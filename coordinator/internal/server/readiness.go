package server

func lifecycleRank(state string) int {
	switch state {
	case "registered":
		return 1
	case "assigned":
		return 2
	case "signaling":
		return 3
	case "load_started":
		return 4
	case "backend_joined":
		return 5
	case "slice_loaded":
		return 6
	case "pipeline_ready", "dry_run_ready":
		return 7
	case "failed":
		return -1
	default:
		return 0
	}
}

func readyLifecycleState(executionMode string) string {
	if executionMode == "dry_run" {
		return "dry_run_ready"
	}
	return "pipeline_ready"
}

func isNodeLifecycleReady(node NodeInfo, executionMode string) bool {
	return node.LifecycleState == readyLifecycleState(executionMode)
}

func backendThresholdRank(executionMode string) int {
	if executionMode == "dry_run" {
		return 0
	}
	return lifecycleRank("backend_joined") // rank 5; single-stage nodes reach slice_loaded (6) which also satisfies this
}

func isBackendReadyNode(node NodeInfo, executionMode string) bool {
	threshold := backendThresholdRank(executionMode)
	if threshold == 0 {
		return false
	}
	return lifecycleRank(node.LifecycleState) >= threshold
}

func isExecutableMode(executionMode string) bool {
	return executionMode != "" && executionMode != "dry_run"
}
