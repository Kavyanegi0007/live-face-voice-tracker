from graph.state import AgentState

# graph/store_node.py


def store_node(state: AgentState) -> AgentState:
    # This node can be used to:
    # - centralize logic
    # - optionally log to file
    # - compute running averages
    # - or just pass through

    # Example (optional): print rolling status
    if state["eye_data"]["total_samples"] % 100 == 0:
        print(f"📊 Eye samples: {state['eye_data']['total_samples']} | "
              f"Blink count: {state['eye_data']['blink_count']}")

    # No modification to state needed for now
    return state
