# graph/build_graph.py

from langgraph.graph import StateGraph
from graph.eye_node import eye_node
from graph.head_pose_node import head_pose_node
from graph.store_node import store_node
#from graph.report_node import report_node  # optional
from graph.state import AgentState
from graph.voice_node import voice_node
def build_graph():
    # Step 1: Create graph
    
    builder = StateGraph(state_schema=AgentState)

    # Step 2: Add nodes
    builder.add_node("eye_node", eye_node)
    builder.add_node("head_pose_node", head_pose_node)
    builder.add_node("voice_node", voice_node)
    builder.add_node("store_node", store_node)

    # Optional node: manual session summary
    #builder.add_node("report_node", report_node)

    # Step 3: Set entry point
    builder.set_entry_point("eye_node")

    # Step 4: Define node flow (eye → head → store → eye again)
    builder.add_edge("eye_node", "head_pose_node")
    builder.add_edge("head_pose_node", "voice_node")
    builder.add_edge("voice_node", "store_node")
    builder.add_edge("store_node", "eye_node")
    # report_node is not wired in the flow — call it manually when needed
    # (e.g., from main.py when user hits "q")

    # Step 5: Compile and return
    return builder.compile()
