# Real Local / Emulated-Cloud Experiment Report

- Runtime config: `real_runtime_emulated_cloud.yaml`
- Edge runtime: local `llama-cpp-python` direct invocation.
- Cloud runtime: OpenAI-compatible API or single-machine multi-worker emulated cloud backend.
- Network profiles are emulated as transport overhead wrapped around real backend inference.
- Note: `mean_quality` is a response-completeness proxy, not a human accuracy score.

## Backend Health

| backend | ok | message | base_url |
| --- | --- | --- | --- |
| edge | True | ok model=Llama-3.2-1B-Instruct-Q4_K_M.gguf | local://llama-cpp |
| cloud | True | ok workers=8 emulated-cloud | emulated://single-machine-cloud |

## Summary

| network | network_label | strategy | strategy_label | request_count | mean_ttft_ms | mean_e2e_ms | p95_e2e_ms | p99_e2e_ms | mean_tpot_ms | mean_throughput_tps | cache_hit_ratio | prefix_hit_ratio | mean_bandwidth_bytes | mean_vram_peak_gb | privacy_exposure_rate | mean_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| congested | Congested | cloud_only | Cloud-only | 24.0 | 390.284 | 612.898 | 1287.086 | 1487.052 | 1.613 | 138.280 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| congested | Congested | edge_only | Edge-only | 24.0 | 421.130 | 598.856 | 633.938 | 748.785 | 2.588 | 104.450 | 0.000 | 0.000 | 547.667 | 3.490 | 0.000 | 0.590 |
| congested | Congested | no_privacy | No-privacy | 24.0 | 393.889 | 623.566 | 1314.142 | 1488.032 | 1.613 | 136.548 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| congested | Congested | ours | Ours | 24.0 | 374.875 | 533.524 | 656.105 | 792.478 | 2.360 | 579.543 | 0.125 | 0.000 | 503.458 | 3.054 | 0.000 | 0.635 |
| mobile_4g5g | 4G/5G | cloud_only | Cloud-only | 24.0 | 368.764 | 571.692 | 1238.718 | 1531.578 | 1.613 | 156.449 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| mobile_4g5g | 4G/5G | edge_only | Edge-only | 24.0 | 413.542 | 580.924 | 622.265 | 639.974 | 2.578 | 106.756 | 0.000 | 0.000 | 554.750 | 3.490 | 0.000 | 0.587 |
| mobile_4g5g | 4G/5G | no_privacy | No-privacy | 24.0 | 378.205 | 586.951 | 1250.972 | 1516.368 | 1.613 | 155.488 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| mobile_4g5g | 4G/5G | ours | Ours | 24.0 | 284.484 | 426.098 | 615.683 | 636.506 | 2.078 | 664.490 | 0.125 | 0.000 | 481.708 | 2.182 | 0.000 | 0.680 |
| weak | Weak | cloud_only | Cloud-only | 24.0 | 436.311 | 715.436 | 1337.653 | 1660.965 | 1.613 | 111.365 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| weak | Weak | edge_only | Edge-only | 24.0 | 432.454 | 625.445 | 654.999 | 824.174 | 2.594 | 101.775 | 0.000 | 0.000 | 550.500 | 3.490 | 0.000 | 0.589 |
| weak | Weak | no_privacy | No-privacy | 24.0 | 420.476 | 686.206 | 1245.359 | 1592.193 | 1.613 | 115.548 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| weak | Weak | ours | Ours | 24.0 | 382.614 | 554.154 | 664.157 | 780.214 | 2.326 | 501.753 | 0.125 | 0.000 | 510.792 | 3.054 | 0.000 | 0.634 |
| wifi_good | WiFi Good | cloud_only | Cloud-only | 24.0 | 361.893 | 541.411 | 1133.735 | 1448.620 | 1.613 | 171.894 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| wifi_good | WiFi Good | edge_only | Edge-only | 24.0 | 402.475 | 560.531 | 599.446 | 603.104 | 2.517 | 111.882 | 0.000 | 0.000 | 552.042 | 3.490 | 0.000 | 0.586 |
| wifi_good | WiFi Good | no_privacy | No-privacy | 24.0 | 358.318 | 532.310 | 1091.823 | 1453.308 | 1.613 | 176.502 | 0.000 | 0.000 | 419.042 | 0.000 | 1.000 | 0.780 |
| wifi_good | WiFi Good | ours | Ours | 24.0 | 274.931 | 409.331 | 610.281 | 613.712 | 2.075 | 671.365 | 0.125 | 0.000 | 475.417 | 2.182 | 0.000 | 0.681 |

## Action Distribution

| network | network_label | route | count | ratio | route_label |
| --- | --- | --- | --- | --- | --- |
| congested | Congested | cache | 3 | 0.125 | Cache |
| congested | Congested | edge | 21 | 0.875 | Edge |
| mobile_4g5g | 4G/5G | cache | 3 | 0.125 | Cache |
| mobile_4g5g | 4G/5G | cloud | 6 | 0.250 | Cloud |
| mobile_4g5g | 4G/5G | edge | 15 | 0.625 | Edge |
| weak | Weak | cache | 3 | 0.125 | Cache |
| weak | Weak | edge | 21 | 0.875 | Edge |
| wifi_good | WiFi Good | cache | 3 | 0.125 | Cache |
| wifi_good | WiFi Good | cloud | 6 | 0.250 | Cloud |
| wifi_good | WiFi Good | edge | 15 | 0.625 | Edge |