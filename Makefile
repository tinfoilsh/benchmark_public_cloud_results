run:
	python3 plot_itl.py
	python3 plot_latency.py
	python3 plot_latency_network.py
	python3 plot_throughput.py

clean:
	rm results_itl.pdf results_e2e_latency.pdf results_ttft_latency.pdf results_network_latency.pdf results_throughput.pdf results_throughput_overhead.pdf