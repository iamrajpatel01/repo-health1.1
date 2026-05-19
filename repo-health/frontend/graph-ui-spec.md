# Antigravity Rules: Graph UI Specification

> **Agent Context:** This rule file is persistent. Refer to these data contracts and visual constraints for all tasks involving graph generation, UI rendering, and mock data filtering.

---

## 1. Mock JSON Data Contract

All graph data must strictly adhere to the following JSON structure. Do not introduce top-level keys outside of `nodes` and `edges`.

```json
{
  "nodes": [
    {
      "id": "string (unique identifier)",
      "label": "string (display name)",
      "group": "string (e.g., 'core', 'periphery')",
      "mass": "number (affects gravitational pull in UI)"
    }
  ],
  "edges": [
    {
      "source": "string (matches a node id)",
      "target": "string (matches a node id)",
      "relation": "string (e.g., 'orbiting', 'repelling')",
      "strength": "number (0.0 to 1.0)"
    }
  ]
}