package server

import (
	"fmt"
	"strings"
)

type StringArrayFlag []string

func (f *StringArrayFlag) String() string {
	return strings.Join(*f, ",")
}

func (f *StringArrayFlag) Set(value string) error {
	*f = append(*f, value)
	return nil
}

type KeyValueFlag map[string]string

func (f *KeyValueFlag) String() string {
	if f == nil {
		return ""
	}
	parts := make([]string, 0, len(*f))
	for key, value := range *f {
		parts = append(parts, fmt.Sprintf("%s=%s", key, value))
	}
	return strings.Join(parts, ",")
}

func (f *KeyValueFlag) Set(value string) error {
	parts := strings.SplitN(value, "=", 2)
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" {
		return fmt.Errorf("expected KEY=VALUE, got %q", value)
	}
	if *f == nil {
		*f = make(map[string]string)
	}
	(*f)[strings.TrimSpace(parts[0])] = parts[1]
	return nil
}

func cloneBackendConfig(cfg BackendConfig) BackendConfig {
	cloned := BackendConfig{
		RayHeadAddress: cfg.RayHeadAddress,
		LaunchArgs:     append([]string(nil), cfg.LaunchArgs...),
	}
	if len(cfg.EnvOverrides) > 0 {
		cloned.EnvOverrides = make(map[string]string, len(cfg.EnvOverrides))
		for key, value := range cfg.EnvOverrides {
			cloned.EnvOverrides[key] = value
		}
	}
	return cloned
}

func IsValidExecutionMode(mode string) bool {
	switch mode {
	case "", "single_node", "slice_loaded_pipeline", "vllm_ray_pipeline", "dry_run":
		return true
	default:
		return false
	}
}
