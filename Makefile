PYTHON ?= python3
SERVICE_SCRIPT := scripts/service.py

.PHONY: service-run service-rollback service-reset

service-run:
	@PYTHONPATH=stubs:src $(PYTHON) $(SERVICE_SCRIPT) run $(ARGS)

service-rollback:
	@PYTHONPATH=stubs:src $(PYTHON) $(SERVICE_SCRIPT) rollback $(ARGS)

service-reset:
	@PYTHONPATH=stubs:src $(PYTHON) $(SERVICE_SCRIPT) reset $(ARGS)
