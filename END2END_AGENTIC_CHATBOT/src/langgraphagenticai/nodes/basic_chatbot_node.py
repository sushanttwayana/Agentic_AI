from src.langgraphagenticai.state.state import State

class BasicChatbotNode:
    """
    Basic Chatbot logic implementation
    """
    
    def __init__(self, model):
        self.model = model
        
    def process(self, state:State) -> dict:
        """
        Processes the unput state and generates a chatbot response

        Args:
            state (State): _description_

        Returns:
            dict: _description_
        """
       
        return {"messages": self.model.invoke(state["messages"])} 