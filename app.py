import streamlit as st
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import re

# Database setup
def init_db():
    conn = sqlite3.connect('llm_recommender.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS llms (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            added_date TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS vote_history (
            id INTEGER PRIMARY KEY,
            task_id INTEGER,
            llm_id INTEGER,
            vote_date TEXT,
            vote_count INTEGER DEFAULT 1,
            source TEXT DEFAULT 'user',  -- 'user', 'reddit', 'web'
            FOREIGN KEY (task_id) REFERENCES tasks (id),
            FOREIGN KEY (llm_id) REFERENCES llms (id)
        )
    ''')
    
    # Initial tasks
    tasks_data = [
        ('Coding',),
        ('Creative Writing',),
        ('Research',),
        ('Summarization',),
        ('Vision Tasks',),
        ('General Conversation',)
    ]
    c.executemany('INSERT OR IGNORE INTO tasks (name) VALUES (?)', tasks_data)
    
    # Initial LLMs (2025 data from Reddit, Artificial Analysis)
    llms_data = [
        ('GPT-5 High (OpenAI)', 'Top for coding, reasoning; IOI/SWE-bench leader.', '2025-09-20'),
        ('Claude 4 Sonnet (Anthropic)', 'Best for writing, ethics; 200K context.', '2025-09-20'),
        ('Gemini 2.5 Pro (Google)', 'Multimodal leader; 1M+ context.', '2025-09-20'),
        ('Grok 4 (xAI)', 'Research, humor; efficient.', '2025-09-20'),
        ('Qwen3 (Alibaba)', 'Coding, local efficiency; 235B params.', '2025-09-20'),
        ('Llama 4 (Meta)', 'Open-source; summarization.', '2025-09-20'),
        ('DeepSeek R1 (DeepSeek)', 'Math, coding; beats o1.', '2025-09-20'),
        ('Mistral Large 2 (Mistral AI)', 'Languages, privacy.', '2025-09-20'),
        ('Kimi K2 (Moonshot AI)', 'Free writing, search.', '2025-09-20'),
        ('GLM-4 (Zhipu AI)', 'Vision, writing.', '2025-09-20'),
        ('Gemma 3 (Google)', 'Local vision, writing; 27B.', '2025-09-20'),
        ('IBM Granite 3.2 (IBM)', 'Enterprise summarization.', '2025-09-20'),
        ('Phi 4 (Microsoft)', 'Compact summarization.', '2025-09-20'),
        ('Command R+ (Cohere)', 'RAG, research.', '2025-09-20')
    ]
    c.executemany('INSERT OR IGNORE INTO llms (name, description, added_date) VALUES (?, ?, ?)', llms_data)
    
    # Initial votes (from 2025 Reddit mentions, e.g., GPT-5 30 for coding)
    initial_votes = [
        (1, 1, '2025-09-20', 30, 'initial'), (1, 7, '2025-09-20', 25, 'initial'), (1, 5, '2025-09-20', 20, 'initial'),
        (2, 3, '2025-09-20', 25, 'initial'), (2, 2, '2025-09-20', 22, 'initial'), (2, 1, '2025-09-20', 20, 'initial'),
        (3, 1, '2025-09-20', 22, 'initial'), (3, 2, '2025-09-20', 20, 'initial'), (3, 5, '2025-09-20', 18, 'initial'),
        (4, 12, '2025-09-20', 20, 'initial'), (4, 8, '2025-09-20', 18, 'initial'), (4, 6, '2025-09-20', 16, 'initial'),
        (5, 3, '2025-09-20', 30, 'initial'), (5, 5, '2025-09-20', 20, 'initial'), (5, 11, '2025-09-20', 18, 'initial'),
        (6, 5, '2025-09-20', 22, 'initial'), (6, 4, '2025-09-20', 20, 'initial'), (6, 2, '2025-09-20', 18, 'initial')
    ]
    c.executemany('INSERT OR IGNORE INTO vote_history (task_id, llm_id, vote_date, vote_count, source) VALUES (?, ?, ?, ?, ?)', initial_votes)
    
    conn.commit()
    conn.close()

# Web scrape for new LLMs
@st.cache_data(ttl=86400)
def fetch_latest_llms():
    try:
        url = "https://artificialanalysis.ai/leaderboards/models"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        models = []
        for row in soup.find_all('tr')[:20]:
            cells = row.find_all('td')
            if len(cells) > 1:
                name = cells[0].text.strip()
                if name and '(' in name:
                    models.append(name)
        conn = sqlite3.connect('llm_recommender.db')
        c = conn.cursor()
        added = 0
        for model in models:
            c.execute('SELECT id FROM llms WHERE name = ?', (model,))
            if c.fetchone() is None:
                desc = f"From Artificial Analysis ({datetime.now().strftime('%Y-%m-%d')})"
                c.execute('INSERT INTO llms (name, description, added_date) VALUES (?, ?, ?)', 
                          (model, desc, datetime.now().strftime('%Y-%m-%d')))
                c.execute('INSERT INTO vote_history (task_id, llm_id, vote_date, vote_count, source) VALUES (?, ?, ?, ?, ?)',
                          (None, c.lastrowid, datetime.now().strftime('%Y-%m-%d'), 0, 'web'))
                added += 1
        conn.commit()
        conn.close()
        return f"Added {added} new LLMs!"
    except Exception as e:
        return f"Web fetch failed: {str(e)}. Using existing data."

# Reddit scrape (public, no API keys)
@st.cache_data(ttl=86400)
def fetch_reddit():
    try:
        tasks_df = get_tasks()
        llms_df = get_llms()
        today = datetime.now().strftime('%Y-%m-%d')
        added = 0
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        for task in tasks_df['name']:
            query = f"best LLM for {task.lower()}".replace(' ', '+')
            url = f"https://www.reddit.com/r/LocalLLaMA/search/?q={query}&sort=new"
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            posts = soup.find_all('a', href=re.compile(r'/r/LocalLLaMA/comments/'))[:5]
            for post in posts:
                post_url = 'https://www.reddit.com' + post['href']
                post_response = requests.get(post_url, headers=headers, timeout=10)
                post_soup = BeautifulSoup(post_response.text, 'html.parser')
                text = (post_soup.find('h1') or '').text.lower() + " "
                body = post_soup.find('div', class_=re.compile(r'rich-text-editor-content.*'))
                text += (body.text if body else '').lower() + " "
                comments = post_soup.find_all('div', class_=re.compile(r'comment.*'))
                for comment in comments:
                    text += (comment.text or '').lower() + " "
                for llm in llms_df['name']:
                    count = len(re.findall(rf'\b{re.escape(llm.lower())}\b', text))
                    if count > 0:
                        add_history_vote(task, llm, count, source='reddit')
                        added += count
        return f"Added {added} Reddit votes!"
    except Exception as e:
        return f"Reddit fetch failed: {str(e)}. Using user votes."

# Data functions
def get_tasks():
    conn = sqlite3.connect('llm_recommender.db')
    tasks = pd.read_sql_query('SELECT * FROM tasks', conn)
    conn.close()
    return tasks

def get_llms():
    conn = sqlite3.connect('llm_recommender.db')
    llms = pd.read_sql_query('SELECT * FROM llms ORDER BY added_date DESC', conn)
    conn.close()
    return llms

def get_votes_for_task(task_name, date=None, daily=True):
    conn = sqlite3.connect('llm_recommender.db')
    query = '''
        SELECT l.name, l.description, SUM(v.vote_count) as total_votes, GROUP_CONCAT(DISTINCT v.source) as sources
        FROM vote_history v
        JOIN tasks t ON v.task_id = t.id
        JOIN llms l ON v.llm_id = l.id
        WHERE t.name = ?
    '''
    params = [task_name]
    if date:
        if daily:
            query += ' AND v.vote_date = ?'
            params.append(date)
        else:
            query += ' AND v.vote_date <= ?'
            params.append(date)
    query += ' GROUP BY l.id ORDER BY total_votes DESC LIMIT 10'
    votes = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()
    return votes

def add_history_vote(task_name, llm_name, count=1, source='user'):
    conn = sqlite3.connect('llm_recommender.db')
    c = conn.cursor()
    if task_name:
        c.execute('SELECT id FROM tasks WHERE name = ?', (task_name,))
        task_row = c.fetchone()
        if not task_row:
            return
        task_id = task_row[0]
    else:
        task_id = None
    c.execute('SELECT id FROM llms WHERE name = ?', (llm_name,))
    llm_row = c.fetchone()
    if not llm_row:
        c.execute('INSERT INTO llms (name, description, added_date) VALUES (?, ?, ?)', 
                  (llm_name, 'User-added', datetime.now().strftime('%Y-%m-%d')))
        llm_id = c.lastrowid
    else:
        llm_id = llm_row[0]
    if task_id:
        c.execute('INSERT INTO vote_history (task_id, llm_id, vote_date, vote_count, source) VALUES (?, ?, ?, ?, ?)',
                  (task_id, llm_id, datetime.now().strftime('%Y-%m-%d'), count, source))
    conn.commit()
    conn.close()

# Streamlit App
def main():
    st.set_page_config(page_title="LLM Recommender 2025", page_icon="🤖", layout="wide")
    init_db()
    
    st.title("🤖 Ultimate LLM Recommender 2025")
    st.markdown("""
    Find the best AI for any task! Rankings from **user surveys** (weighted by experience), **Reddit mentions** (r/LocalLLaMA), 
    and **daily web scrapes** (Artificial Analysis). Add new AIs, vote, and check daily leaderboards—no setup needed!
    """)
    
    # Sidebar: Add LLM only
    st.sidebar.header("Add New LLM")
    new_llm = st.sidebar.text_input("Name (e.g., NewModel (Company))")
    new_desc = st.sidebar.text_area("Description")
    if st.sidebar.button("Add LLM") and new_llm:
        add_history_vote(None, new_llm, 0)
        st.sidebar.success("Added! Vote in Survey tab.")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Survey", "Leaderboards"])
    
    with tab1:
        st.header("Recommendations")
        tasks = get_tasks()['name'].tolist()
        selected_task = st.selectbox("What task do you need?", tasks)
        votes_df = get_votes_for_task(selected_task)
        if not votes_df.empty:
            for idx, row in votes_df.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{row['name']}** - {row['description']}")
                    st.caption(f"Sources: {row['sources']}")
                with col2:
                    if st.button(f"👍 Vote", key=f"rec_{idx}"):
                        add_history_vote(selected_task, row['name'])
                        st.rerun()
                with col3:
                    st.metric("Total Votes", row['total_votes'])
        else:
            st.warning("No votes yet. Add/vote in Survey!")
    
    with tab2:
        st.header("Survey - Weight Your Vote")
        with st.form("survey_form"):
            num_ais = st.number_input("How many LLMs have you used?", min_value=0, max_value=100, value=1)
            st.caption("Votes weighted by experience (e.g., 5 LLMs = 5x vote weight).")
            preferences = {}
            llm_names = get_llms()['name'].tolist()
            for task in tasks:
                preferences[task] = st.selectbox(f"Best LLM for {task}", ["None"] + llm_names)
            if st.form_submit_button("Submit Survey"):
                for task, llm in preferences.items():
                    if llm != "None":
                        add_history_vote(task, llm, count=num_ais)
                st.success(f"Votes added with {num_ais}x weight!")
                st.rerun()
    
    with tab3:
        st.header("Leaderboards - Daily & All-Time")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("All-Time Rankings")
            for task in tasks:
                with st.expander(task):
                    df = get_votes_for_task(task, daily=False)
                    df['Sources'] = df['sources'].apply(lambda x: x.replace(',', ', '))
                    st.dataframe(df[['name', 'total_votes', 'Sources']], use_container_width=True)
        with col2:
            st.subheader("Daily Rankings")
            sel_date = st.date_input("Select Date", value=date.today(), max_value=date.today())
            date_str = sel_date.strftime('%Y-%m-%d')
            for task in tasks:
                with st.expander(task):
                    df = get_votes_for_task(task, date_str, daily=True)
                    df['Sources'] = df['sources'].apply(lambda x: x.replace(',', ', '))
                    st.dataframe(df[['name', 'total_votes', 'Sources']], use_container_width=True)
    
    # Updates
    st.header("Keep It Fresh")
    colu1, colu2 = st.columns(2)
    with colu1:
        if st.button("🔄 Refresh Web Leaderboards"):
            result = fetch_latest_llms()
            st.info(result)
            st.rerun()
    with colu2:
        if st.button("🔄 Update from Reddit"):
            result = fetch_reddit()
            st.info(result)
            st.rerun()
    
    # Transparency
    with st.expander("How It Works"):
        st.markdown("""
        - **User Votes**: Weighted by # of LLMs used in Survey.
        - **Reddit**: Daily mentions from r/LocalLLaMA posts/comments.
        - **Web**: Daily scrapes from Artificial Analysis leaderboard.
        - **Initial Data**: 2025 Reddit polls, benchmark mentions.
        - **No Bias**: Purely vote/mention-driven. Add new AIs anytime!
        """)
    
    st.markdown("---")
    st.markdown("Built by Prakhar Joshi | Sep 20, 2025 ")

if __name__ == "__main__":
    main()