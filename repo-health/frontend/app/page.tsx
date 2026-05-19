"use client";

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import ReactFlow, { 
  Background, 
  Controls, 
  MiniMap, 
  useNodesState, 
  useEdgesState, 
  BackgroundVariant, 
  ReactFlowProvider,
  Handle,
  Position,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  Node,
  Edge
} from 'reactflow';
import 'reactflow/dist/style.css';
import { AlertTriangle, GitMerge, FileWarning, TrendingDown, Cpu, Activity, Bot, Search, GitBranch, X, CheckCircle } from 'lucide-react';
import dagre from 'dagre';

// --- Types ---
interface HealthMetrics {
  overall_score: number;
  complexity_total: number;
  dependency_cycles: number;
}
interface TimelineCommit {
  commit_hash: string;
  timestamp: string;
  author: string;
  health_metrics: HealthMetrics;
}
interface GraphState {
  nodes: any[];
  edges: any[];
}
interface LLMTrigger {
  anomaly_detected: boolean;
  trigger_reason: string;
  explanation?: string;
  git_diff_snippet?: string;
}
interface CommitDetails {
  commit_hash: string;
  timestamp: string;
  author: string;
  health_metrics: HealthMetrics;
  graph_state: GraphState;
  llm_trigger_payload: LLMTrigger;
}

// --- Custom Node ---
const CustomNode = ({ data }: any) => {
  const isHighRisk = data.complexity_score > 15;
  const isDrift = data.architectural_drift;
  const isBusFactorCritical = data.bus_factor === 1;

  const borderColor = isDrift ? 'border-[#FF3B5C]' : isBusFactorCritical ? 'border-[#FF8C00]' : isHighRisk ? 'border-[#FF8C00]' : 'border-[#3B82F6]';
  const bgColor = isDrift ? 'bg-[#FF3B5C]/10' : isBusFactorCritical ? 'bg-[#FF8C00]/10' : isHighRisk ? 'bg-[#FF8C00]/10' : 'bg-[#3B82F6]/10';

  return (
    <div className={`px-4 py-2 shadow-xl rounded-md border ${borderColor} ${bgColor} backdrop-blur-md min-w-[120px]`}>
      <Handle type="target" position={Position.Top} className="w-2 h-2 bg-slate-400" />
      <div className="flex flex-col gap-0.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-bold text-white truncate">{data.label}</span>
          {isBusFactorCritical && (
            <span className="text-[8px] font-bold px-1 py-0.5 rounded bg-[#FF8C00]/30 text-[#FF8C00] border border-[#FF8C00]/50 shrink-0" title="Single author owns >80% of this module">
              BF:1
            </span>
          )}
        </div>
        <span className="text-[10px] text-slate-300 font-mono">Cx: {data.complexity_score}</span>
      </div>
      <Handle type="source" position={Position.Bottom} className="w-2 h-2 bg-slate-400" />
    </div>
  );
};


// --- Custom Edge ---
const ToxicEdge = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {}, markerEnd }: any) => {
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />
      <path d={edgePath} fill="none" stroke="#FF3B5C" strokeWidth={4} className="animate-pulse opacity-30" />
    </>
  );
};

// --- Layout Engine ---
const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 150, height: 50 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = Position.Top;
    node.sourcePosition = Position.Bottom;
    node.position = {
      x: nodeWithPosition.x - 75,
      y: nodeWithPosition.y - 25,
    };
    return node;
  });

  return { nodes, edges };
};

// --- Main Application ---
const nodeTypes = { default: CustomNode };
const edgeTypes = { toxic: ToxicEdge };

export default function RepoHealthDashboard() {
  const [timelineData, setTimelineData] = useState<TimelineCommit[]>([]);
  const [selectedCommitId, setSelectedCommitId] = useState<string>('');
  const [commitDetails, setCommitDetails] = useState<CommitDetails | null>(null);
  const [loadingTimeline, setLoadingTimeline] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // --- Analyze Repo state ---
  const [repoUrl, setRepoUrl] = useState('');
  const [analyzeStatus, setAnalyzeStatus] = useState<'idle'|'cloning'|'analyzing'|'done'|'error'>('idle');
  const [analyzeMessage, setAnalyzeMessage] = useState('');
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const fetchTimeline = useCallback(() => {
    fetch('http://localhost:8000/api/timeline')
      .then(res => res.json())
      .then((data: TimelineCommit[]) => {
        setTimelineData(data);
        if (data.length > 0) setSelectedCommitId(data[data.length - 1].commit_hash);
        setLoadingTimeline(false);
      })
      .catch(err => { console.error('Failed to load timeline', err); setLoadingTimeline(false); });
  }, []);

  const startAnalysis = useCallback(() => {
    if (!repoUrl.trim()) return;
    setAnalyzeStatus('cloning');
    setAnalyzeMessage('Starting analysis...');
    fetch('http://localhost:8000/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_url: repoUrl.trim(), max_commits: 600 }),
    })
    .then(res => { if (!res.ok) return res.json().then(e => Promise.reject(e.detail)); return res.json(); })
    .then(() => {
      // Start polling
      pollRef.current = setInterval(() => {
        fetch('http://localhost:8000/api/analyze/status')
          .then(r => r.json())
          .then(s => {
            setAnalyzeStatus(s.status);
            setAnalyzeMessage(s.message);
            if (s.status === 'done') {
              clearInterval(pollRef.current!);
              fetchTimeline();  // reload the dashboard
            }
            if (s.status === 'error') clearInterval(pollRef.current!);
          });
      }, 2000);
    })
    .catch(err => { setAnalyzeStatus('error'); setAnalyzeMessage(String(err)); });
  }, [repoUrl, fetchTimeline]);

  // 1. Fetch Timeline on mount
  useEffect(() => { fetchTimeline(); }, [fetchTimeline]);
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // 2. Fetch Commit Details
  useEffect(() => {
    if (!selectedCommitId) return;
    setLoadingDetails(true);
    fetch(`http://localhost:8000/api/commit/${selectedCommitId}`)
      .then(res => res.json())
      .then((data: CommitDetails) => { setCommitDetails(data); setLoadingDetails(false); })
      .catch(err => { console.error('Failed to load commit details', err); setLoadingDetails(false); });
  }, [selectedCommitId]);

  const isAnalyzing = analyzeStatus === 'cloning' || analyzeStatus === 'analyzing';

  if (loadingTimeline && analyzeStatus === 'idle') {
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-950 text-white font-mono">
        <div className="flex items-center gap-3">
          <div className="w-4 h-4 bg-[#3B82F6] animate-ping rounded-full" />
          <span className="text-sm tracking-widest text-slate-400">INITIALIZING INGESTION PIPELINE...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen bg-neutral-950 text-slate-200 overflow-hidden font-sans">
      
      {/* Top Navbar */}
      <header className="shrink-0 border-b border-white/10 bg-neutral-900/50 backdrop-blur-md z-10">
        {/* Row 1: branding + commit count */}
        <div className="h-14 flex items-center px-6 justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-[#3B82F6]" />
            <h1 className="text-sm font-bold tracking-widest uppercase text-white">Repo Health Intelligence</h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="px-3 py-1.5 rounded-md bg-white/5 border border-white/10 text-xs font-mono text-slate-400">
              {timelineData.length} Commits Analyzed
            </div>
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#3B82F6] to-[#A855F7] flex items-center justify-center text-xs font-bold text-white shadow-lg">
              HQ
            </div>
          </div>
        </div>

        {/* Row 2: Analyze Repo input */}
        <div className="h-12 flex items-center px-6 gap-3 border-t border-white/5">
          <GitBranch className="w-4 h-4 text-slate-500 shrink-0" />
          <input
            id="repo-url-input"
            type="text"
            placeholder="Paste a public GitHub URL (e.g. https://github.com/langgenius/dify)"
            value={repoUrl}
            onChange={e => setRepoUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !isAnalyzing && startAnalysis()}
            disabled={isAnalyzing}
            className="flex-1 bg-transparent text-xs font-mono text-slate-300 placeholder-slate-600 outline-none disabled:opacity-50"
          />
          {analyzeMessage && (
            <span className={`text-[10px] font-mono truncate max-w-xs ${
              analyzeStatus === 'error' ? 'text-[#FF3B5C]' :
              analyzeStatus === 'done'  ? 'text-[#00FF88]' : 'text-[#FF8C00]'
            }`}>
              {analyzeStatus === 'done' && <CheckCircle className="w-3 h-3 inline mr-1" />}
              {analyzeMessage}
            </span>
          )}
          <button
            id="analyze-btn"
            onClick={startAnalysis}
            disabled={isAnalyzing || !repoUrl.trim()}
            className="shrink-0 flex items-center gap-2 px-4 py-1.5 rounded-lg bg-[#3B82F6] hover:bg-[#2563EB] disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-bold transition-all"
          >
            {isAnalyzing ? (
              <><Activity className="w-3 h-3 animate-spin" />{analyzeStatus === 'cloning' ? 'Cloning...' : 'Analyzing...'}</>
            ) : (
              <><Search className="w-3 h-3" />Analyze Repo</>
            )}
          </button>
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Left Column (30%): KPIs & Timeline */}
        <div className="w-[30%] shrink-0 flex flex-col border-r border-white/10 bg-neutral-900/30 overflow-hidden">
          
          {/* KPIs */}
          <div className="p-5 flex flex-col gap-4 shrink-0 border-b border-white/10">
            <h2 className="text-xs font-bold tracking-widest uppercase text-slate-500 mb-2">Architectural Telemetry</h2>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="p-4 rounded-xl bg-white/5 border border-white/10 relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-16 h-16 bg-[#3B82F6]/10 rounded-full blur-2xl group-hover:bg-[#3B82F6]/20 transition-all" />
                <div className="flex items-center gap-2 mb-2">
                  <TrendingDown className="w-4 h-4 text-[#3B82F6]" />
                  <span className="text-[10px] uppercase tracking-wider text-slate-400">Health Score</span>
                </div>
                <div className="text-3xl font-light font-mono text-white">
                  {commitDetails?.health_metrics.overall_score.toFixed(1) || '--'}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-white/5 border border-white/10 relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-16 h-16 bg-[#FF8C00]/10 rounded-full blur-2xl group-hover:bg-[#FF8C00]/20 transition-all" />
                <div className="flex items-center gap-2 mb-2">
                  <Cpu className="w-4 h-4 text-[#FF8C00]" />
                  <span className="text-[10px] uppercase tracking-wider text-slate-400">Complexity</span>
                </div>
                <div className="text-3xl font-light font-mono text-white">
                  {commitDetails?.health_metrics.complexity_total || 0}
                </div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-white/5 border border-white/10 relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-16 h-16 bg-[#FF3B5C]/10 rounded-full blur-2xl group-hover:bg-[#FF3B5C]/20 transition-all" />
              <div className="flex items-center gap-2 mb-2">
                <GitMerge className="w-4 h-4 text-[#FF3B5C]" />
                <span className="text-[10px] uppercase tracking-wider text-slate-400">Dependency Cycles</span>
              </div>
              <div className="text-3xl font-light font-mono text-white">
                {commitDetails?.health_metrics.dependency_cycles || 0}
              </div>
            </div>
          </div>

          {/* Recharts Timeline */}
          <div className="flex-1 p-5 flex flex-col min-h-0">
            <h2 className="text-xs font-bold tracking-widest uppercase text-slate-500 mb-4 shrink-0">VIX Drift Index</h2>
            <div className="w-full h-[300px] shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                  data={timelineData} 
                  onClick={(e: any) => {
                    if (e?.activePayload?.[0]?.payload) {
                      setSelectedCommitId(e.activePayload[0].payload.commit_hash);
                    }
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis 
                    dataKey="commit_hash" 
                    tickFormatter={(hash) => hash.substring(0, 6)}
                    stroke="rgba(255,255,255,0.2)"
                    tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10, fontFamily: 'monospace' }}
                    minTickGap={30}
                  />
                  <YAxis 
                    domain={['auto', 100]} 
                    stroke="rgba(255,255,255,0.2)"
                    tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10, fontFamily: 'monospace' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0a0a0a', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                    itemStyle={{ color: '#fff', fontFamily: 'monospace' }}
                    labelStyle={{ color: '#888', marginBottom: '4px' }}
                    formatter={(val: number) => [val.toFixed(1), 'Score']}
                    labelFormatter={(label) => `Commit: ${label.substring(0, 8)}`}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="health_metrics.overall_score" 
                    stroke="#3B82F6" 
                    strokeWidth={2}
                    dot={(props: any) => {
                      const isSelected = props.payload.commit_hash === selectedCommitId;
                      return (
                        <circle 
                          cx={props.cx} cy={props.cy} r={isSelected ? 6 : 0} 
                          fill={isSelected ? '#fff' : 'transparent'} 
                          stroke="#3B82F6" strokeWidth={2} 
                          style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                        />
                      );
                    }}
                    activeDot={{ r: 6, fill: '#3B82F6', stroke: '#fff', strokeWidth: 2, cursor: 'pointer' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 text-[10px] text-slate-500 text-center font-mono shrink-0">
              CLICK A NODE ON THE CHART TO INSPECT COMMIT TOPOLOGY
            </div>
          </div>
        </div>

        {/* Center Column (45%): React Flow */}
        <div className="w-[45%] shrink-0 flex flex-col border-r border-white/10 bg-[#020202] relative">
          <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-neutral-900/80 backdrop-blur-md px-4 py-2 rounded-lg border border-white/10 shadow-2xl">
            <div className={`w-2 h-2 rounded-full ${loadingDetails ? 'bg-[#FF8C00] animate-pulse' : 'bg-[#00FF88]'}`} />
            <span className="text-[10px] font-mono font-bold tracking-widest text-white uppercase">
              {loadingDetails ? 'Parsing Topology...' : `Commit: ${selectedCommitId.substring(0, 8)}`}
            </span>
          </div>

          <div className="flex-1 w-full h-full">
            <ReactFlowProvider>
              <FlowCanvas graphState={commitDetails?.graph_state} />
            </ReactFlowProvider>
          </div>
        </div>

        {/* Right Column (25%): AI Risk & Hotspots */}
        <div className="flex-1 flex flex-col bg-neutral-900/30 overflow-y-auto">
          {commitDetails && <SidebarContent commitDetails={commitDetails} />}
        </div>

      </div>
    </div>
  );
}

// --- React Flow Canvas Component ---
function FlowCanvas({ graphState }: { graphState?: GraphState }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    if (!graphState) return;

    // Map raw nodes to React Flow format with strict styling rules based on complexity
    const mappedNodes: Node[] = graphState.nodes.map(n => ({
      id: n.id,
      type: 'default',
      position: n.position || { x: Math.random() * 500, y: Math.random() * 500 }, // Fallback if no position
      data: {
        label: n.data?.label || n.id,
        complexity_score: n.data?.complexity_score || 0,
        architectural_drift: n.data?.complexity_score > 15
      }
    }));

    // Map edges, giving violation edges a red dashed style
    const mappedEdges: Edge[] = graphState.edges.map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: e.is_violation ? 'toxic' : 'default',
      animated: e.is_violation,
      style: { stroke: e.is_violation ? '#FF3B5C' : '#334155', strokeWidth: e.is_violation ? 2 : 1 },
    }));

    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(mappedNodes, mappedEdges);

    setNodes(layoutedNodes);
    setEdges(layoutedEdges);
  }, [graphState, setNodes, setEdges]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      fitView
      fitViewOptions={{ padding: 0.2, maxZoom: 1.5 }}
      minZoom={0.1}
      maxZoom={2}
      proOptions={{ hideAttribution: true }}
      className="bg-[#020202]"
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="rgba(255,255,255,0.05)" />
      <Controls className="bg-neutral-900 border border-white/10 fill-white" showInteractive={false} />
    </ReactFlow>
  );
}

// --- Sidebar Content Component ---
function SidebarContent({ commitDetails }: { commitDetails: CommitDetails }) {
  const [llmExplanation, setLlmExplanation] = useState<string | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);

  // Listen for anomalies and trigger the LLM
  useEffect(() => {
    const triggerPayload = commitDetails?.llm_trigger_payload;
    
    if (triggerPayload?.anomaly_detected && triggerPayload?.git_diff_snippet) {
      setIsExplaining(true);
      
      fetch('http://localhost:8000/api/explain-anomaly', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          git_diff_snippet: triggerPayload.git_diff_snippet,
          trigger_reason: triggerPayload.trigger_reason,
          topological_delta: triggerPayload.topological_delta || null,
        })
      })
      .then(res => res.json())
      .then(data => setLlmExplanation(data.explanation))
      .catch(err => console.error("LLM Error:", err))
      .finally(() => setIsExplaining(false));
    } else {
      setLlmExplanation(null); // Reset if no anomaly
    }
  }, [commitDetails]);

  const isAnomaly = commitDetails.llm_trigger_payload?.anomaly_detected;
  
  // Dynamic calculation of top hotspots
  const topHotspots = useMemo(() => {
    const n = commitDetails.graph_state?.nodes || [];
    return [...n]
      .sort((a, b) => (b.data?.complexity_score || 0) - (a.data?.complexity_score || 0))
      .slice(0, 5)
      .map(n => ({ name: n.data?.label || n.id, score: n.data?.complexity_score || 0 }));
  }, [commitDetails.graph_state]);

  // Dynamic calculation of cycles
  const circularDependencies = useMemo(() => {
    const e = commitDetails.graph_state?.edges || [];
    return e.filter(edge => edge.is_violation).map(edge => ({
      source: edge.source.split('/').pop() || 'Unknown',
      target: edge.target.split('/').pop() || 'Unknown'
    }));
  }, [commitDetails.graph_state]);

  return (
    <div className="p-5 flex flex-col gap-6">
      
      {/* AI Risk Analysis */}
      <div className="flex flex-col gap-3">
        <h2 className="text-xs font-bold tracking-widest uppercase text-slate-500 flex items-center gap-2">
          <Bot className="w-4 h-4 text-[#A855F7]" />
          Gemini Risk Analysis
        </h2>
        
        {isAnomaly ? (
          <div className="p-4 rounded-xl bg-[rgba(255,59,92,0.05)] border border-[#FF3B5C]/30 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-24 h-24 bg-[#FF3B5C]/10 rounded-full blur-3xl" />
            <div className="flex items-center gap-2 mb-3 relative z-10">
              <AlertTriangle className="w-4 h-4 text-[#FF3B5C] animate-pulse" />
              <span className="text-[11px] font-bold tracking-wider text-[#FF3B5C] uppercase">
                Critical Drift Detected
              </span>
            </div>
            <p className="text-sm text-slate-200 leading-relaxed relative z-10">
              {isExplaining ? (
                <span className="flex items-center gap-2 text-slate-400 italic">
                  <Activity className="w-3 h-3 animate-spin" /> Analyzing structural decay...
                </span>
              ) : (
                llmExplanation || commitDetails.llm_trigger_payload.trigger_reason
              )}
            </p>
          </div>
        ) : (
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <p className="text-sm text-slate-400 italic">
              "No critical architectural anomalies detected."
            </p>
          </div>
        )}
      </div>

      {/* Top Hotspots */}
      <div className="flex flex-col gap-3">
        <h2 className="text-xs font-bold tracking-widest uppercase text-slate-500 flex items-center gap-2">
          <FileWarning className="w-4 h-4 text-[#FF8C00]" />
          Complexity Hotspots
        </h2>
        <div className="flex flex-col gap-2">
          {topHotspots.length > 0 ? topHotspots.map((h, i) => (
            <div key={i} className="flex items-center gap-3 p-2.5 rounded-lg bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
              <span className="text-[10px] text-slate-500 font-mono w-4 shrink-0">{(i + 1).toString().padStart(2, '0')}</span>
              <span className="flex-1 text-sm text-slate-200 font-mono truncate">{h.name}</span>
              <span className={`text-xs font-bold font-mono px-2 py-0.5 rounded ${h.score > 15 ? 'bg-[#FF8C00]/20 text-[#FF8C00]' : 'bg-[#3B82F6]/20 text-[#3B82F6]'}`}>
                {h.score}
              </span>
            </div>
          )) : (
            <div className="text-xs text-slate-500 italic p-2">No files analyzed.</div>
          )}
        </div>
      </div>

      {/* Circular Dependencies */}
      <div className="flex flex-col gap-3">
        <h2 className="text-xs font-bold tracking-widest uppercase text-slate-500 flex items-center gap-2">
          <GitMerge className="w-4 h-4 text-[#FF3B5C]" />
          Active Cycles
        </h2>
        <div className="flex flex-col gap-2">
          {circularDependencies.length > 0 ? circularDependencies.map((c, i) => (
            <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-[rgba(255,59,92,0.05)] border border-[#FF3B5C]/20 hover:border-[#FF3B5C]/40 transition-colors">
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <span className="text-xs font-mono text-slate-300 truncate">{c.source}</span>
                <span className="text-[#FF3B5C] shrink-0">↔</span>
                <span className="text-xs font-mono text-slate-300 truncate">{c.target}</span>
              </div>
            </div>
          )) : (
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <span className="text-xs text-[#00FF88] font-mono uppercase tracking-widest">Topology Clean</span>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
