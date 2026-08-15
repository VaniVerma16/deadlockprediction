.PHONY: workloads test demo clean

workloads:
	$(MAKE) -C workloads

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m deadlock_dataset.cli build \
		--events tests/fixtures/abba_events.jsonl \
		--output build/demo-snapshots.jsonl \
		--interval-ms 10 --unsafe-window-ms 30 --confirm-ms 20
	PYTHONPATH=src python3 -m deadlock_dataset.cli validate \
		--dataset build/demo-snapshots.jsonl

clean:
	$(MAKE) -C workloads clean
	rm -rf build

