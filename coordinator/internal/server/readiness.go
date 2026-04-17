package server

func lifecycleRank(state string) int {
	switch state {
	case "registered":
		return 1
	case "assigned":
		return 2
	case "load_started":
		return 3
	case "backend_joined":
		return 4
	case "slice_loaded":
		return 5
	case "pipeline_ready", "dry_run_ready":
		return 6
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
	switch executionMode {
	case "dry_run":
		return 0
	case "single_node":
		return lifecycleRank("slice_loaded")
	default:
		return lifecycleRank("backend_joined")
	}
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
