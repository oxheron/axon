package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"axon/coordinator/internal/server"
)

func main() {
	host := flag.String("host", "0.0.0.0", "Coordinator bind host")
	port := flag.Int("port", 8000, "Coordinator bind port")
	minNodes := flag.Int("min-nodes", 2, "Minimum nodes required before startup signal")
	modelName := flag.String("model-name", "", "Model name/path used by all nodes and the user service")
	executionMode := flag.String("execution-mode", "", "Explicit execution mode override: vllm_slice, coordinator_slice, or dry_run")
	clusterToken := flag.String("cluster-token", "", "Pre-shared token nodes must supply to open a WebSocket session (empty = no auth)")
	signalTimeout := flag.Duration("signal-timeout", 60*time.Second, "Deadline for all nodes to send their signal after the first signal arrives")
	var backendLaunchArgs server.StringArrayFlag
	var backendEnv server.KeyValueFlag
	flag.Var(&backendLaunchArgs, "backend-launch-arg", "Repeatable backend launch argument appended to node worker startup.")
	flag.Var(&backendEnv, "backend-env", "Repeatable backend environment override in KEY=VALUE form.")
	flag.Parse()

	if *modelName == "" {
		fmt.Fprintln(os.Stderr, "--model-name is required")
		os.Exit(2)
	}
	if *minNodes < 1 {
		fmt.Fprintln(os.Stderr, "--min-nodes must be >= 1")
		os.Exit(2)
	}
	if !server.IsValidExecutionMode(*executionMode) {
		fmt.Fprintf(os.Stderr, "invalid --execution-mode %q\n", *executionMode)
		os.Exit(2)
	}

	srv := server.NewServer(
		*minNodes,
		*modelName,
		*executionMode,
		server.BackendConfig{
			EnvOverrides: backendEnv,
			LaunchArgs:   backendLaunchArgs,
		},
	)
	srv.SetSignalingOptions(*clusterToken, *signalTimeout)

	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("starting coordinator on %s", addr)
	log.Fatal(http.ListenAndServe(addr, srv.Handler()))
}
