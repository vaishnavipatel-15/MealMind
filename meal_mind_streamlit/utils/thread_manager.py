import streamlit as st
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import json

class ThreadManager:
    """Manages conversation threads for users"""
    
    def __init__(self, conn):
        self.conn = conn
    
    def create_thread(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation thread"""
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        
        # Auto-generate title if not provided
        if not title:
            title = f"Conversation {datetime.now().strftime('%b %d, %I:%M %p')}"
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO conversation_threads 
                (thread_id, user_id, title, created_at, last_message_at, message_count, is_active)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 0, TRUE)
            """, (thread_id, user_id, title))
            
            self.conn.commit()
            return thread_id
        except Exception as e:
            st.error(f"Error creating thread: {e}")
            return None
        finally:
            cursor.close()
    
    def get_user_threads(self, user_id: str, limit: int = 3) -> List[Dict]:
        """Get user's recent threads"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT thread_id, title, created_at, last_message_at, message_count, summary
                FROM conversation_threads
                WHERE user_id = %s AND is_active = TRUE
                ORDER BY last_message_at DESC
                LIMIT %s
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            threads = []
            for row in rows:
                threads.append({
                    'thread_id': row[0],
                    'title': row[1],
                    'created_at': row[2],
                    'last_message_at': row[3],
                    'message_count': row[4],
                    'summary': row[5]
                })
            return threads
        except Exception as e:
            st.error(f"Error fetching threads: {e}")
            return []
        finally:
            cursor.close()
    
    def add_message(self, thread_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a message to a thread"""
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        cursor = self.conn.cursor()
        try:
            # Insert message - handle metadata properly
            if metadata:
                metadata_json = json.dumps(metadata)
                cursor.execute("""
                    INSERT INTO thread_messages 
                    (message_id, thread_id, role, content, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP(), PARSE_JSON(%s))
                """, (message_id, thread_id, role, content, metadata_json))
            else:
                cursor.execute("""
                    INSERT INTO thread_messages 
                    (message_id, thread_id, role, content, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP(), NULL)
                """, (message_id, thread_id, role, content))
            
            # Update thread metadata
            cursor.execute("""
                UPDATE conversation_threads
                SET last_message_at = CURRENT_TIMESTAMP(),
                    message_count = message_count + 1
                WHERE thread_id = %s
            """, (thread_id,))
            
            self.conn.commit()
            return message_id
        except Exception as e:
            st.error(f"Error adding message: {e}")
            return None
        finally:
            cursor.close()
    
    def get_thread_messages(self, thread_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get messages from a thread"""
        cursor = self.conn.cursor()
        try:
            query = """
                SELECT message_id, role, content, timestamp, metadata
                FROM thread_messages
                WHERE thread_id = %s
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (thread_id,))
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                messages.append({
                    'message_id': row[0],
                    'role': row[1],
                    'content': row[2],
                    'timestamp': row[3],
                    'metadata': row[4] if row[4] else {}
                })
            return messages
        except Exception as e:
            st.error(f"Error fetching messages: {e}")
            return []
        finally:
            cursor.close()
    
    def update_thread_title(self, thread_id: str, title: str):
        """Update thread title"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE conversation_threads
                SET title = %s
                WHERE thread_id = %s
            """, (title, thread_id))
            self.conn.commit()
        except Exception as e:
            st.error(f"Error updating thread title: {e}")
        finally:
            cursor.close()
    
    def generate_thread_title(self, thread_id: str, first_message: str, use_llm: bool = True) -> str:
        """Generate a title from the first message using LLM or simple heuristic"""
        
        if use_llm:
            try:
                from langchain_community.chat_models import ChatSnowflakeCortex
                from langchain.schema import SystemMessage, HumanMessage
                from utils.db import get_snowpark_session
                
                session = get_snowpark_session()
                llm = ChatSnowflakeCortex(session=session, model="llama3.1-70b")
                
                system_prompt = """Generate a concise, descriptive title (max 6 words) for this conversation based on the first message.
The title should capture the main topic or question.

Examples:
- "What's for breakfast today?" → "Breakfast Plan Inquiry"
- "I want to lose 10 pounds" → "Weight Loss Goal"
- "Tell me about Italian recipes" → "Italian Recipe Exploration"

Return ONLY the title, nothing else."""
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"First message: {first_message[:200]}")
                ]
                
                response = llm.invoke(messages)
                title = response.content.strip().strip('"')
                
                # Update the thread with new title
                self.update_thread_title(thread_id, title)
                return title
                
            except Exception as e:
                print(f"LLM title generation failed: {e}, falling back to simple method")
        
        # Fallback: Simple heuristic
        if '?' in first_message[:100]:
            title = first_message.split('?')[0] + '?'
        else:
            title = first_message[:50]
        
        if len(first_message) > 50:
            title += "..."
        
        self.update_thread_title(thread_id, title)
        return title
    
    def archive_thread(self, thread_id: str):
        """Archive a thread (soft delete)"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE conversation_threads
                SET is_active = FALSE
                WHERE thread_id = %s
            """, (thread_id,))
            self.conn.commit()
        except Exception as e:
            st.error(f"Error archiving thread: {e}")
        finally:
            cursor.close()
    
    def summarize_thread(self, thread_id: str, summary: str):
        """Save thread summary"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE conversation_threads
                SET summary = %s
                WHERE thread_id = %s
            """, (summary, thread_id))
            self.conn.commit()
        except Exception as e:
            st.error(f"Error saving summary: {e}")
        finally:
            cursor.close()


class ThreadMemoryManager:
    """Manages short-term and long-term memory for threads"""
    
    def __init__(self, conn, thread_id: str):
        self.conn = conn
        self.thread_id = thread_id
        self.thread_manager = ThreadManager(conn)
    
    def get_conversation_context(self, last_n: int = 10) -> List[Dict]:
        """Get recent conversation context (short-term memory)"""
        return self.thread_manager.get_thread_messages(self.thread_id, limit=last_n)
    
    def save_checkpoint(self, state_data: Dict):
        """Save LangGraph state checkpoint"""
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
        
        cursor = self.conn.cursor()
        try:
            state_json = json.dumps(state_data)
            cursor.execute("""
                INSERT INTO thread_checkpoints
                (checkpoint_id, thread_id, checkpoint_data, created_at)
                VALUES (%s, %s, PARSE_JSON(%s), CURRENT_TIMESTAMP())
            """, (checkpoint_id, self.thread_id, state_json))
            self.conn.commit()
            return checkpoint_id
        except Exception as e:
            st.error(f"Error saving checkpoint: {e}")
            return None
        finally:
            cursor.close()
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the latest checkpoint for this thread"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT checkpoint_data
                FROM thread_checkpoints
                WHERE thread_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (self.thread_id,))
            
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]  # Snowflake VARIANT returns as dict
            return None
        except Exception as e:
            st.error(f"Error loading checkpoint: {e}")
            return None
        finally:
            cursor.close()
