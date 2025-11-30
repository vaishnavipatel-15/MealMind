import streamlit as st
from typing import Dict, Any, List, TypedDict, Optional, Literal
from langchain_community.chat_models import ChatSnowflakeCortex
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import json
import re
import os
import requests
from datetime import datetime

# ==================== LANGGRAPH STATE ====================
class ChatRouterState(TypedDict):
    user_input: str
    user_id: str
    user_profile: Dict
    inventory_summary: str
    meal_plan_summary: str
    history: List[Any]
    route: Optional[Literal["meal_retrieval", "general_chat", "calorie_estimation", "meal_adjustment", "inventory_modification"]]
    retrieved_data: Optional[str]
    user_preferences: Optional[Dict]
    extracted_feedback: Optional[List[Dict]]
    response: str
    adjustment_result: Optional[Dict]
    monitoring_warnings: Optional[List[str]]
    inventory_result: Optional[Dict]
    inventory_response: Optional[str]

# ==================== MULTI-AGENT ROUTER ====================
class MealRouterAgent:
    """Intelligent routing agent that directs queries to specialized agents"""
    
    def __init__(self, session, conn):
        self.session = session
        self.conn = conn
        try:
            self.chat_model = ChatSnowflakeCortex(
                session=self.session,
                model="llama3.1-70b",
                cortex_search_service="MEAL_MIND"
            )
        except Exception as e:
            st.warning(f"Chat Model initialization failed: {e}")
            self.chat_model = None
        
        from utils.feedback_agent import FeedbackAgent
        from utils.meal_adjustment_agent import MealAdjustmentAgent
        from utils.monitoring_agent import MonitoringAgent
        
        self.feedback_agent = FeedbackAgent(conn, session)
        self.adjustment_agent = MealAdjustmentAgent(session, conn)
        self.monitoring_agent = MonitoringAgent(conn)
    
    # ==================== LOAD PREFERENCES NODE ====================
    def node_load_preferences(self, state: ChatRouterState) -> ChatRouterState:
        """Load user preferences from long-term memory"""
        preferences = self.feedback_agent.get_user_preferences(state['user_id'])
        state['user_preferences'] = preferences
        return state
    
    # ==================== EXTRACT FEEDBACK NODE ====================
    def node_extract_feedback(self, state: ChatRouterState) -> ChatRouterState:
        """Extract preferences from user message"""
        extracted = self.feedback_agent.extract_preferences(
            state['user_input'], 
            state['user_id']
        )
        state['extracted_feedback'] = extracted
        return state
    
    def node_route_query(self, state: ChatRouterState) -> ChatRouterState:
        """Analyze user query and determine which agent to route to"""
        user_input = state['user_input'].lower()
        
        # Keywords that indicate MODIFICATION of inventory
        inventory_modification_keywords = [
            'bought', 'added', 'got', 'purchased', 'add', 'buy',
            'used', 'consumed', 'ate', 'removed', 'delete', 'remove',
            'swap', 'replace', 'substitute', 'exchange', 'update',
            'drank', 'drunk', 'finished'
        ]
        
        # Keywords that indicate VIEWING inventory
        inventory_view_keywords = [
            'what', 'show', 'list', 'see', 'view', 'check', 'tell me'
        ]
        
        meal_keywords = [
            'meal', 'recipe', 'breakfast', 'lunch', 'dinner', 'snack',
            'what am i eating', 'what should i eat', 'meal plan',
            'today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 
            'thursday', 'friday', 'saturday', 'sunday',
            'ingredient', 'what can i make', 'show me', 'get me'
        ]
        
        estimation_keywords = [
            'estimate', 'calculate', 'how many calories in', 'nutritional info for'
        ]
        
        adjustment_keywords = [
            'change', 'replace', 'swap', 'instead', 'don\'t want', 'ate', 'had', 'went to', 
            'buffet', 'restaurant', 'eaten', 'drank', 'consumed'
        ]
        
        is_adjustment_intent = False
        if any(keyword in user_input for keyword in adjustment_keywords) and \
           any(meal in user_input for meal in ['breakfast', 'lunch', 'dinner', 'snack', 'meal']):
            is_adjustment_intent = True
            
            if 'what' in user_input and not any(k in user_input for k in ['change', 'replace', 'swap', 'add', 'instead']):
                is_adjustment_intent = False
        
        # Check for inventory intent
        has_inventory_word = 'inventory' in user_input or 'stock' in user_input or 'ingredients' in user_input
        is_modification = any(keyword in user_input for keyword in inventory_modification_keywords)
        is_viewing = any(keyword in user_input for keyword in inventory_view_keywords)
        
        # If it mentions inventory AND has modification keywords (but NOT viewing keywords), route to modification
        if has_inventory_word and is_modification and not is_viewing:
            state['route'] = 'inventory_modification'
            return state
        
        # If it mentions inventory with viewing keywords, route to general_chat
        if has_inventory_word and is_viewing:
            state['route'] = 'general_chat'
            return state
        
        # NEW: If it has modification keywords WITHOUT meal context, assume inventory modification
        # (e.g., "I bought apples", "I added chicken")
        if is_modification and not any(meal in user_input for meal in ['breakfast', 'lunch', 'dinner', 'snack', 'meal']):
            state['route'] = 'inventory_modification'
            return state

        if is_adjustment_intent:
            state['route'] = 'meal_adjustment'
        elif any(keyword in user_input for keyword in estimation_keywords) and not ('my plan' in user_input or 'my meal' in user_input):
            state['route'] = 'calorie_estimation'
        elif any(keyword in user_input for keyword in meal_keywords):
            state['route'] = 'meal_retrieval'
        else:
            state['route'] = 'general_chat'
        
        return state
    
    # ==================== MEAL RETRIEVAL NODE ====================
    def node_retrieve_meals(self, state: ChatRouterState) -> ChatRouterState:
        """Retrieve meal information from database based on user query"""
        from utils.db import get_meals_by_criteria, get_meal_details_by_type
        
        user_input = state['user_input'].lower()
        user_id = state['user_id']
        
        try:
            meal_type = None
            if 'breakfast' in user_input:
                meal_type = 'breakfast'
            elif 'lunch' in user_input:
                meal_type = 'lunch'
            elif 'dinner' in user_input:
                meal_type = 'dinner'
            elif 'snack' in user_input:
                meal_type = 'snacks'
            
            days_map = {
                'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4,
                'friday': 5, 'saturday': 6, 'sunday': 7,
                'today': datetime.now().weekday() + 1
            }
            
            day_number = None
            for day_name, day_num in days_map.items():
                if day_name in user_input:
                    day_number = day_num
                    break
            
            meals = get_meals_by_criteria(self.conn, user_id, day_number, meal_type)
            
            if meals:
                formatted_data = "## Retrieved Meals\n\n"
                for meal in meals:
                    formatted_data += f"**{meal.get('meal_name', 'Unknown')}** ({meal.get('meal_type', '').title()})\n"
                    formatted_data += f"- Day: {meal.get('day_name', 'Unknown')}\n"
                    
                    nutrition = meal.get('nutrition', {})
                    if nutrition:
                        formatted_data += f"- Calories: {nutrition.get('calories', 'N/A')} kcal\n"
                        formatted_data += f"- Protein: {nutrition.get('protein_g', 'N/A')}g\n"
                    
                    ingredients = meal.get('ingredients_with_quantities', [])
                    if ingredients:
                        formatted_data += "- Ingredients: " + ", ".join([ing.get('ingredient', '') for ing in ingredients[:5]]) + "\n"
                    
                    formatted_data += "\n"
                
                state['retrieved_data'] = formatted_data
            else:
                state['retrieved_data'] = "No meals found matching your criteria. You may not have an active meal plan."
        
        except Exception as e:
            state['retrieved_data'] = f"Error retrieving meals: {str(e)}"
        
        return state
    
    # ==================== MEAL ADJUSTMENT NODE ====================
    def node_adjust_meal(self, state: ChatRouterState) -> ChatRouterState:
        """Handle meal changes and restaurant entries"""
        user_input = state['user_input'].lower()
        user_id = state['user_id']
        
        meal_type = 'lunch'
        if 'breakfast' in user_input: meal_type = 'breakfast'
        elif 'dinner' in user_input: meal_type = 'dinner'
        elif 'snack' in user_input: meal_type = 'snacks'
        
        date = datetime.now().strftime('%Y-%m-%d')
        
        result = self.adjustment_agent.process_request(
            user_input, user_id, date, meal_type
        )
        
        state['adjustment_result'] = result
        return state

    # ==================== INVENTORY MODIFICATION NODE ====================
    def node_modify_inventory(self, state: ChatRouterState) -> ChatRouterState:
        """Handle inventory modifications via LLM parsing and database updates"""
        from utils.helpers import add_inventory_item, update_inventory_quantity
        
        user_input = state['user_input']
        user_id = state['user_id']
        
        try:
            # Use LLM to parse the user's intent and extract inventory items
            system_prompt = """You are an inventory management assistant. Parse the user's message and extract inventory modifications.

Return a JSON object with this structure:
{
    "action": "add" or "remove" or "swap",
    "items": [
        {
            "name": "item name",
            "quantity": number,
            "unit": "unit (kg, g, pieces, etc.)",
            "category": "category (Produce, Dairy, Meat, etc.)"
        }
    ],
    "swap_from": "old item name" (only for swap action),
    "swap_to": "new item name" (only for swap action)
}

Examples:
- "I bought 2kg of apples" -> {"action": "add", "items": [{"name": "apples", "quantity": 2, "unit": "kg", "category": "Produce"}]}
- "I used 500g of chicken" -> {"action": "remove", "items": [{"name": "chicken", "quantity": 500, "unit": "g", "category": "Meat"}]}
- "I ate 3 apples" -> {"action": "remove", "items": [{"name": "apples", "quantity": 3, "unit": "pieces", "category": "Produce"}]}
- "swap oranges for mandarin oranges" -> {"action": "swap", "swap_from": "oranges", "swap_to": "mandarin oranges", "items": [{"name": "mandarin oranges", "quantity": 1, "unit": "pieces", "category": "Produce"}]}

Return ONLY the JSON object, no other text."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            
            if not self.chat_model:
                state['inventory_response'] = "⚠️ Chat model not available. Cannot process inventory modification."
                return state
            
            # Get LLM response
            response = self.chat_model.invoke(messages)
            response_text = response.content.strip()
            
            # Parse JSON from response
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                parsed_data = json.loads(response_text)
            
            action = parsed_data.get('action', 'add')
            items = parsed_data.get('items', [])
            swap_from = parsed_data.get('swap_from')
            swap_to = parsed_data.get('swap_to')
            
            if not items and action != 'swap':
                state['inventory_response'] = "❌ Could not understand what items to modify. Please be more specific."
                return state
            
            # Process each item
            success_messages = []
            failed_items = []
            
            if action == 'swap' and swap_from and swap_to:
                # Handle swap: remove old item, add new item
                # First, try to find and remove the old item
                cursor = self.conn.cursor()
                try:
                    cursor.execute("""
                        SELECT inventory_id, quantity, unit, category
                        FROM inventory
                        WHERE user_id = %s AND LOWER(item_name) = LOWER(%s)
                        LIMIT 1
                    """, (user_id, swap_from))
                    
                    result = cursor.fetchone()
                    
                    if result:
                        old_inventory_id, old_quantity, old_unit, old_category = result
                        
                        # Delete the old item
                        cursor.execute("DELETE FROM inventory WHERE inventory_id = %s", (old_inventory_id,))
                        
                        # Add the new item with same quantity
                        new_item = items[0] if items else {}
                        new_quantity = new_item.get('quantity', old_quantity)
                        new_unit = new_item.get('unit', old_unit)
                        new_category = new_item.get('category', old_category)
                        
                        success = add_inventory_item(
                            self.conn,
                            user_id,
                            swap_to,
                            new_quantity,
                            new_unit,
                            new_category
                        )
                        
                        if success:
                            success_messages.append(f"Swapped {swap_from} for {swap_to} ({new_quantity} {new_unit})")
                        else:
                            failed_items.append(f"Failed to add {swap_to}")
                    else:
                        failed_items.append(f"{swap_from} not found in inventory")
                        
                except Exception as e:
                    failed_items.append(f"Swap error: {str(e)}")
                finally:
                    cursor.close()
            else:
                # Handle regular add/remove
                for item in items:
                    item_name = item.get('name')
                    quantity = item.get('quantity', 1)
                    unit = item.get('unit', 'pieces')
                    category = item.get('category', 'Other')
                    
                    if action == 'add':
                        # Add to inventory
                        success = add_inventory_item(
                            self.conn, 
                            user_id, 
                            item_name, 
                            quantity, 
                            unit, 
                            category
                        )
                        if success:
                            success_messages.append(f"Added {quantity} {unit} of {item_name}")
                        else:
                            failed_items.append(item_name)
                    
                    elif action == 'remove':
                        # Remove from inventory (reduce quantity)
                        success, new_qty, message = update_inventory_quantity(
                            self.conn,
                            user_id,
                            item_name,
                            -quantity  # Negative to subtract
                        )
                        if success:
                            success_messages.append(message)
                        else:
                            failed_items.append(f"{item_name} ({message})")
            
            # Generate response
            if success_messages:
                state['inventory_response'] = "✅ " + "\n".join(success_messages)
                state['inventory_result'] = {'status': 'success', 'items_modified': len(success_messages)}
            elif failed_items:
                state['inventory_response'] = f"❌ Failed to modify items: {', '.join(failed_items)}"
                state['inventory_result'] = {'status': 'error', 'message': 'Database error'}
            else:
                state['inventory_response'] = "❌ No items were modified."
                state['inventory_result'] = {'status': 'error', 'message': 'No items processed'}
                
        except json.JSONDecodeError as e:
            state['inventory_response'] = f"❌ Error parsing inventory data: {str(e)}"
            state['inventory_result'] = {'status': 'error', 'message': str(e)}
        except Exception as e:
            state['inventory_response'] = f"❌ Error updating inventory: {str(e)}"
            state['inventory_result'] = {'status': 'error', 'message': str(e)}
        
        return state

    # ==================== MONITORING NODE ====================
    def node_monitor_changes(self, state: ChatRouterState) -> ChatRouterState:
        """Monitor changes and generate warnings"""
        if state.get('adjustment_result', {}).get('status') == 'success':
            user_id = state['user_id']
            date = datetime.now().strftime('%Y-%m-%d')
            
            warnings = self.monitoring_agent.monitor_changes(user_id, date)
            state['monitoring_warnings'] = warnings
            
        return state

    # ==================== GENERAL CHAT NODE ====================
    def node_general_chat(self, state: ChatRouterState) -> ChatRouterState:
        """Handle general nutrition and cooking questions"""
        user_profile = state['user_profile']
        inventory = state['inventory_summary']
        meal_plan = state['meal_plan_summary']
        history = state['history']
        preferences = state.get('user_preferences', {})
        
        pref_text = self.feedback_agent.format_preferences_for_prompt(preferences)
        
        system_prompt = f"""You are Meal Mind AI, a helpful nutrition and meal planning assistant.

USER PROFILE:
- Name: {user_profile.get('username', 'User')}
- Goal: {user_profile.get('health_goal', 'General Health')}
- Dietary Restrictions: {user_profile.get('dietary_restrictions', 'None')}
- Allergies: {user_profile.get('food_allergies', 'None')}

USER PREFERENCES (LEARNED):
{pref_text}

CURRENT INVENTORY:
{inventory[:500]}...

MEAL PLAN SUMMARY:
{meal_plan[:300]}...

YOUR ROLE:
- Provide nutrition advice and cooking tips considering user preferences
- Answer health and wellness questions
- Be encouraging and supportive
- Keep responses concise and helpful
- IMPORTANT: Respect user dislikes and preferences in your suggestions
"""
        
        messages = [SystemMessage(content=system_prompt)]
        
        recent_history = history[-5:]
        start_index = 0
        
        for i, msg in enumerate(recent_history):
            if isinstance(msg, HumanMessage):
                start_index = i
                break
            if i == len(recent_history) - 1:
                start_index = len(recent_history)
        
        for msg in recent_history[start_index:]:
            messages.append(msg)
        
        messages.append(HumanMessage(content=state['user_input']))
        
        try:
            if self.chat_model:
                response = self.chat_model.invoke(messages)
                state['response'] = response.content
            else:
                state['response'] = "I'm currently in offline mode. Please check your connection."
        except Exception as e:
            state['response'] = f"I encountered an error: {str(e)}"
        
        return state

    # ==================== CALORIE ESTIMATION NODE ====================
    def node_estimate_calories(self, state: ChatRouterState) -> ChatRouterState:
        """Estimate calories for unstructured food descriptions"""
        user_input = state['user_input']
        
        system_prompt = """You are an expert nutritionist and calorie estimator. 
The user will describe a meal (e.g., from a buffet, restaurant, or home cooking).

Your task is to:
1. Analyze the food items described.
2. Estimate portion sizes if not specified (make reasonable assumptions based on standard servings).
3. Calculate the approximate Calories and Macronutrients (Protein, Carbs, Fat) for each item and the total.
4. Provide a clear breakdown.
5. Offer a brief, non-judgmental health tip regarding this meal.

Format the output using Markdown:
- Use bold for totals.
- Use a list for the breakdown.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        try:
            if self.chat_model:
                response = self.chat_model.invoke(messages)
                state['response'] = response.content
            else:
                state['response'] = "I'm offline and cannot estimate calories right now."
        except Exception as e:
            state['response'] = f"Error estimating calories: {str(e)}"
            
        return state
    
    # ==================== RESPONSE GENERATION NODE ====================
    def node_generate_response(self, state: ChatRouterState) -> ChatRouterState:
        """Generate final response using retrieved data if available"""
        
        # Case 1: Inventory Modification Result
        if state.get('inventory_response'):
            state['response'] = state['inventory_response']
            return state
        
        # Case 2: Adjustment Result
        if state.get('adjustment_result'):
            result = state['adjustment_result']
            warnings = state.get('monitoring_warnings', [])
            
            if result['status'] == 'success':
                response = f"✅ {result['message']}\n\n"
                response += "**New Daily Total:**\n"
                totals = result['new_daily_total']
                response += f"- Calories: {totals['calories']} kcal\n"
                response += f"- Protein: {totals['protein']}g\n"
                response += f"- Carbs: {totals['carbohydrates']}g\n"
                response += f"- Fat: {totals['fat']}g\n"
                response += f"- Fiber: {totals['fiber']}g\n"
                
                if warnings:
                    response += "\n**Health Alerts:**\n"
                    for w in warnings:
                        response += f"{w}\n"
            else:
                response = f"❌ {result['message']}"
                
            state['response'] = response
            return state

        # Case 3: Retrieved Meal Data
        if state.get('retrieved_data'):
            user_profile = state['user_profile']
            
            system_prompt = f"""You are Meal Mind AI. The user asked about their meals and we retrieved this data:

{state['retrieved_data']}

USER PROFILE:
- Goal: {user_profile.get('health_goal', 'General Health')}

Generate a helpful, conversational response that:
1. Presents the meal information clearly
2. Relates it to their health goals
3. Offers any relevant tips or suggestions
4. Keep it concise and friendly
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state['user_input'])
            ]
            
            try:
                if self.chat_model:
                    response = self.chat_model.invoke(messages)
                    state['response'] = response.content
                else:
                    state['response'] = state['retrieved_data']
            except Exception as e:
                state['response'] = state['retrieved_data']
        
        return state
    
    # ==================== CONDITIONAL EDGES ====================
    def should_retrieve_meals(self, state: ChatRouterState) -> str:
        """Determine next node based on route"""
        if state['route'] == 'meal_retrieval':
            return 'retrieve_meals'
        elif state['route'] == 'calorie_estimation':
            return 'estimate_calories'
        elif state['route'] == 'meal_adjustment':
            return 'adjust_meal'
        elif state['route'] == 'inventory_modification':
            return 'modify_inventory'
        else:
            return 'general_chat'
    
    # ==================== BUILD GRAPH ====================
    def build_graph(self):
        """Build the LangGraph workflow with memory integration"""
        workflow = StateGraph(ChatRouterState)
        
        workflow.add_node("load_preferences", self.node_load_preferences)
        workflow.add_node("extract_feedback", self.node_extract_feedback)
        workflow.add_node("route_query", self.node_route_query)
        workflow.add_node("retrieve_meals", self.node_retrieve_meals)
        workflow.add_node("estimate_calories", self.node_estimate_calories)
        workflow.add_node("adjust_meal", self.node_adjust_meal)
        workflow.add_node("modify_inventory", self.node_modify_inventory)
        workflow.add_node("monitor_changes", self.node_monitor_changes)
        workflow.add_node("general_chat", self.node_general_chat)
        workflow.add_node("generate_response", self.node_generate_response)
        
        workflow.set_entry_point("load_preferences")
        workflow.add_edge("load_preferences", "extract_feedback")
        workflow.add_edge("extract_feedback", "route_query")
        
        workflow.add_conditional_edges(
            "route_query",
            self.should_retrieve_meals,
            {
                "retrieve_meals": "retrieve_meals",
                "estimate_calories": "estimate_calories",
                "adjust_meal": "adjust_meal",
                "modify_inventory": "modify_inventory",
                "general_chat": "general_chat"
            }
        )
        
        workflow.add_edge("adjust_meal", "monitor_changes")
        workflow.add_edge("monitor_changes", "generate_response")
        workflow.add_edge("modify_inventory", "generate_response")
        workflow.add_edge("retrieve_meals", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("general_chat", END)
        workflow.add_edge("estimate_calories", END)
        
        return workflow.compile()
    
    # ==================== RUN METHODS ====================
    def run_chat(self, user_input: str, user_id: str, history: List[Any], context_data: Dict) -> str:
        """Main entry point to run the multi-agent chat"""
        
        initial_state = ChatRouterState(
            user_input=user_input,
            user_id=user_id,
            user_profile=context_data.get('user_profile', {}),
            inventory_summary=context_data.get('inventory_summary', ''),
            meal_plan_summary=context_data.get('meal_plan_summary', ''),
            history=history,
            route=None,
            retrieved_data=None,
            user_preferences=None,
            extracted_feedback=None,
            response="",
            adjustment_result=None,
            monitoring_warnings=None,
            inventory_result=None,
            inventory_response=None
        )
        
        app = self.build_graph()
        result = app.invoke(initial_state)
        
        return result['response']
    
    def run_chat_stream(self, user_input: str, user_id: str, history: List[Any], context_data: Dict):
        """Stream the chat response word by word"""
        
        full_response = self.run_chat(user_input, user_id, history, context_data)
        
        words = full_response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
