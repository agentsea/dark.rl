
.PHONY: test-message
test-message:
	uv run python integration_test_message_queue.py

.PHONY: install-node
install-node:
	./install_node.sh