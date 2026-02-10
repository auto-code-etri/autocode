import sys
import os

# Create outputs directory
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Add ArchCode to python path for prompt templates
sys.path.append(os.path.abspath("third_party/etri_langgraph"))

from networks.archcode.archcode import ArchNet

def main():
    print("Initializing ArchNet...")
    net = ArchNet()
    
    print("Compiling Graph...")
    graph = net.compile()
    
    print("Graph Compiled Successfully.")
    print("Graph Object:", graph)
    
    # Optional: Print graph structure if available
    if hasattr(graph, "get_graph"):
        try:
            gra = graph.get_graph()
            print("Nodes:", gra.nodes)
            print("Edges:", gra.edges)
        except Exception as e:
            print("Could not visualize graph structure:", e)
            
    print("\nVerification Complete: ArchNet migrated to etri_langgraph successfully (Compilation only).")

if __name__ == "__main__":
    main()
