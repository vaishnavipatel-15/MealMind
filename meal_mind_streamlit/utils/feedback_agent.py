import streamlit as st
import uuid
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from langchain_community.chat_models import ChatSnowflakeCortex
from langchain.schema import SystemMessage, HumanMessage

class FeedbackAgent:
    """Intelligent agent that extracts and tracks user preferences from conversations"""
    
    def __init__(self, conn, session):
        self.conn = conn
        self.session = session
        
        try:
            self.llm = ChatSnowflakeCortex(
                session=self.session,
                model="llama3.1-70b"
            )
        except Exception as e:
            st.warning(f"Feedback Agent LLM init failed: {e}")
            self.llm = None
    
    def extract_preferences(self, user_message: str, user_id: str) -> List[Dict]:
        """Extract preferences from user message using LLM"""
        
        if not self.llm:
            return []
        
        system_prompt = """You are a food preference extraction expert. Analyze the user's message and extract any food-related preferences, likes, dislikes, or dietary requests.

Examples:
- "I love salmon" → {"type": "like", "entity": "salmon", "entity_type": "ingredient", "sentiment": "positive", "intensity": 5}
- "Not a fan of mushrooms" → {"type": "dislike", "entity": "mushrooms", "entity_type": "ingredient", "sentiment": "negative", "intensity": 3}
- "Next week I want Italian food" → {"type": "temporal_preference", "entity": "italian", "entity_type": "cuisine", "timing": "next_week"}
- "I'm trying to reduce carbs" → {"type": "dietary_goal", "entity": "low_carb", "entity_type": "dietary"}
- "I want more protein" → {"type": "macro_preference", "entity": "protein", "value": "increase"}

Return ONLY a valid JSON array of preferences. If no preferences found, return empty array [].
Each preference must have: type, entity, entity_type, and optionally sentiment, intensity (1-5), timing, or value."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User message: \"{user_message}\"\n\nExtract preferences:")
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Try to parse JSON
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            preferences = json.loads(content)
            
            # Store each preference
            for pref in preferences:
                self.save_feedback(
                    user_id=user_id,
                    feedback_type=pref.get('type', 'preference'),
                    entity_type=pref.get('entity_type', 'unknown'),
                    entity_name=pref.get('entity'),
                    sentiment=pref.get('sentiment', 'neutral'),
                    intensity=pref.get('intensity', 3),
                    context=user_message,
                    source='inferred',
                    metadata=pref
                )
            
            return preferences
            
        except Exception as e:
            print(f"Preference extraction error: {e}")
            return []
    
    def save_feedback(
        self, 
        user_id: str, 
        feedback_type: str,
        entity_type: str,
        entity_name: str,
        sentiment: str = 'neutral',
        intensity: int = 3,
        context: str = '',
        source: str = 'explicit',
        metadata: Optional[Dict] = None
    ):
        """Save user feedback to database"""
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO user_feedback
                (feedback_id, user_id, feedback_type, entity_type, entity_id, entity_name,
                 sentiment, intensity, context, extracted_at, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP(), %s)
            """, (
                feedback_id, user_id, feedback_type, entity_type,
                metadata.get('entity_id') if metadata else None,
                entity_name, sentiment, intensity, context, source
            ))
            
            self.conn.commit()
            
            # Also update or create preference
            self.update_preference(user_id, feedback_type, entity_type, entity_name, metadata)
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
        finally:
            cursor.close()
    
    def update_preference(
        self, 
        user_id: str, 
        pref_type: str, 
        entity_type: str, 
        entity_name: str,
        metadata: Optional[Dict] = None
    ):
        """Update or create user preference in long-term memory"""
        
        cursor = self.conn.cursor()
        try:
            # Check if preference exists
            cursor.execute("""
                SELECT preference_id, frequency, confidence_score
                FROM user_preferences
                WHERE user_id = %s 
                AND preference_type = %s 
                AND preference_key = %s
                AND is_active = TRUE
            """, (user_id, pref_type, entity_name))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing preference
                pref_id, freq, confidence = existing
                new_freq = freq + 1
                new_confidence = min(1.0, confidence + 0.1)  # Increase confidence
                
                cursor.execute("""
                    UPDATE user_preferences
                    SET frequency = %s,
                        confidence_score = %s,
                        last_mentioned = CURRENT_TIMESTAMP(),
                        preference_value = %s
                    WHERE preference_id = %s
                """, (new_freq, new_confidence, entity_type, pref_id))
            else:
                # Create new preference
                pref_id = f"pref_{uuid.uuid4().hex[:12]}"
                
                # Set expiry for temporal preferences
                expires_at = None
                if metadata and 'timing' in metadata:
                    if metadata['timing'] == 'next_week':
                        expires_at = datetime.now() + timedelta(days=7)
                    elif metadata['timing'] == 'this_month':
                        expires_at = datetime.now() + timedelta(days=30)
                
                cursor.execute("""
                    INSERT INTO user_preferences
                    (preference_id, user_id, preference_type, preference_key, 
                     preference_value, confidence_score, frequency, last_mentioned, 
                     expires_at, is_active)
                    VALUES (%s, %s, %s, %s, %s, 0.7, 1, CURRENT_TIMESTAMP(), %s, TRUE)
                """, (pref_id, user_id, pref_type, entity_name, entity_type, expires_at))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"Error updating preference: {e}")
        finally:
            cursor.close()
    
    def get_user_preferences(self, user_id: str) -> Dict[str, List]:
        """Retrieve all active user preferences (long-term memory)"""
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT preference_type, preference_key, preference_value, 
                       confidence_score, frequency, last_mentioned
                FROM user_preferences
                WHERE user_id = %s 
                AND is_active = TRUE
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP())
                ORDER BY frequency DESC, last_mentioned DESC
            """, (user_id,))
            
            rows = cursor.fetchall()
            
            # Organize by type
            preferences = {
                'likes': [],
                'dislikes': [],
                'cuisines': [],
                'dietary': [],
                'temporal': [],
                'other': []
            }
            
            for row in rows:
                pref_type, key, value, confidence, freq, last_mentioned = row
                pref_data = {
                    'name': key,
                    'type': value,
                    'confidence': confidence,
                    'frequency': freq,
                    'last_mentioned': last_mentioned
                }
                
                if pref_type == 'like':
                    preferences['likes'].append(pref_data)
                elif pref_type == 'dislike':
                    preferences['dislikes'].append(pref_data)
                elif pref_type == 'temporal_preference' and value == 'cuisine':
                    preferences['cuisines'].append(pref_data)
                elif 'dietary' in pref_type:
                    preferences['dietary'].append(pref_data)
                elif 'temporal' in pref_type:
                    preferences['temporal'].append(pref_data)
                else:
                    preferences['other'].append(pref_data)
            
            return preferences
            
        except Exception as e:
            print(f"Error fetching preferences: {e}")
            return {}
        finally:
            cursor.close()
    
    def format_preferences_for_prompt(self, preferences: Dict) -> str:
        """Format preferences for inclusion in LLM prompt"""
        
        prompt = ""
        
        if preferences.get('likes'):
            likes = [p['name'] for p in preferences['likes'][:5]]
            prompt += f"User Likes: {', '.join(likes)}\n"
        
        if preferences.get('dislikes'):
            dislikes = [p['name'] for p in preferences['dislikes'][:5]]
            prompt += f"User Dislikes (AVOID): {', '.join(dislikes)}\n"
        
        if preferences.get('cuisines'):
            cuisines = [p['name'] for p in preferences['cuisines'][:3]]
            prompt += f"Preferred Cuisines: {', '.join(cuisines)}\n"
        
        if preferences.get('dietary'):
            dietary = [p['name'] for p in preferences['dietary']]
            prompt += f"Dietary Preferences: {', '.join(dietary)}\n"
        
        if preferences.get('temporal'):
            temporal = preferences['temporal'][0] if preferences['temporal'] else None
            if temporal:
                prompt += f"Current Request: {temporal['name']} ({temporal.get('type', '')})\n"
        
        return prompt if prompt else "No specific preferences recorded."
    
    def save_explicit_feedback(self, user_id: str, entity_id: str, entity_name: str, 
                               entity_type: str, feedback: str):
        """Save explicit thumbs up/down feedback"""
        
        sentiment = 'positive' if feedback == 'like' else 'negative'
        intensity = 5 if feedback == 'like' else 1
        
        self.save_feedback(
            user_id=user_id,
            feedback_type=feedback,
            entity_type=entity_type,
            entity_name=entity_name,
            sentiment=sentiment,
            intensity=intensity,
            context=f"Explicit {feedback} on {entity_name}",
            source='thumbs_up' if feedback == 'like' else 'thumbs_down',
            metadata={'entity_id': entity_id}
        )
