package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"

	"axon/coordinator/internal/server"
)

func detectLocalIP() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()

	localAddr, ok := conn.LocalAddr().(*net.UDPAddr)
	if !ok {
		return "127.0.0.1"
	}
	return localAddr.IP.String()
}

func maybeStartRayHead(enabled bool, port int) {
	if !enabled {
		return
	}

	cmd := exec.Command("ray", "start", "--head", "--port", fmt.Sprintf("%d", port), "--disable-usage-stats")
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("ray head start exited with error (this may be expected if already running): %v: %s", err, string(output))
		return
	}
	log.Printf("ray head started successfully")
}

func main() {
	host := flag.String("host", "0.0.0.0", "Coordinator bind host")
	port := flag.Int("port", 8000, "Coordinator bind port")
	minNodes := flag.Int("min-nodes", 2, "Minimum nodes required before startup signal")
	modelName := flag.String("model-name", "", "Model name/path used by all nodes and the user service")
	executionMode := flag.String("execution-mode", "", "Explicit execution mode override: single_node, slice_loaded_pipeline, vllm_ray_pipeline, or dry_run")
	rayPort := flag.Int("ray-port", 6379, "Ray GCS port for head node")
	rayHeadAddress := flag.String("ray-head-address", "", "Ray head address host:port. If omitted, inferred from local IP + --ray-port")
	autostartRayHead := flag.Bool("autostart-ray-head", false, "Start `ray start --head` before serving HTTP API")
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

	maybeStartRayHead(*autostartRayHead, *rayPort)

	resolvedRayHead := *rayHeadAddress
	if resolvedRayHead == "" {
		resolvedRayHead = fmt.Sprintf("%s:%d", detectLocalIP(), *rayPort)
	}

	srv := server.NewServer(
		*minNodes,
		*modelName,
		resolvedRayHead,
		*executionMode,
		server.BackendConfig{
			RayHeadAddress: resolvedRayHead,
			EnvOverrides:   backendEnv,
			LaunchArgs:     backendLaunchArgs,
		},
	)
	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("starting coordinator on %s", addr)
	log.Fatal(http.ListenAndServe(addr, srv.Handler()))
}
