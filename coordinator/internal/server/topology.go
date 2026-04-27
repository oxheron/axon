package server

func buildPeerNodes(nodes []NodeInfo, stageIndex int) []PeerNode {
	peers := make([]PeerNode, 0, 2)
	if stageIndex > 0 {
		prev := nodes[stageIndex-1]
		peers = append(peers, PeerNode{
			NodeID:     prev.NodeID,
			StageIndex: stageIndex - 1,
			StageRole:  prev.StageRole,
			WorkerURL:  prev.WorkerURL,
		})
	}
	if stageIndex+1 < len(nodes) {
		next := nodes[stageIndex+1]
		peers = append(peers, PeerNode{
			NodeID:     next.NodeID,
			StageIndex: stageIndex + 1,
			StageRole:  next.StageRole,
			WorkerURL:  next.WorkerURL,
		})
	}
	return peers
}

func defaultExecutionMode(_ int) string {
	return "vllm_slice"
}

func stageRoleForIndex(index, stageCount int) string {
	if stageCount <= 1 || index == 0 {
		return "entry"
	}
	if index == stageCount-1 {
		return "final"
	}
	return "middle"
}

func loadStrategyForMode(executionMode string) string {
	switch executionMode {
	case "coordinator_slice":
		return "coordinator_slice"
	case "dry_run":
		return "dry_run"
	default:
		return "vllm_run"
	}
}
