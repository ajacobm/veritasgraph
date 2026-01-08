import gradio as gr
import os
import asyncio
import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Graph visualization
from graph_visualizer import (
    create_graph_html_for_query,
    get_graph_stats,
    load_graph_data,
    extract_entities_from_response
)

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Import OpenAI-compatible API configuration from separate module
from openai_config import get_api_type, get_llm_config, get_embedding_config

# Load .env from the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))
join = os.path.join

PRESET_MAPPING = {
    "Default": {
        "community_level": 2,
        "response_type": "Multiple Paragraphs"
    },
    "Detailed": {
        "community_level": 4,
        "response_type": "Multi-Page Report"
    },
    "Quick": {
        "community_level": 1,
        "response_type": "Single Paragraph"
    },
    "Bullet": {
        "community_level": 2,
        "response_type": "List of 3-7 Points"
    },
    "Comprehensive": {
        "community_level": 5,
        "response_type": "Multi-Page Report"
    },
    "High-Level": {
        "community_level": 1,
        "response_type": "Single Page"
    },
    "Focused": {
        "community_level": 3,
        "response_type": "Multiple Paragraphs"
    }
}

async def global_search(query, input_dir, community_level=2, temperature=0.5, response_type="Multiple Paragraphs"):
        llm_config = get_llm_config()

        llm = ChatOpenAI(
            api_key=llm_config["api_key"],
            api_base=llm_config["api_base"],
            model=llm_config["model"],
            api_type=llm_config["api_type"],
            max_retries=llm_config["max_retries"],
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        
        entity_df = pd.read_parquet(join(input_dir, f"{ENTITY_TABLE}.parquet"))
        report_df = pd.read_parquet(join(input_dir, f"{COMMUNITY_REPORT_TABLE}.parquet"))
        entity_embedding_df = pd.read_parquet(join(input_dir, f"{ENTITY_EMBEDDING_TABLE}.parquet"))

        reports = read_indexer_reports(report_df, entity_df, community_level)
        entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,
            token_encoder=token_encoder,
        )

        context_builder_params = {
            "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 4000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            "context_name": "Reports",
        }

        map_llm_params = {
            "max_tokens": 1000,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        reduce_llm_params = {
            "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
            "temperature": temperature,
        }

        search_engine = GlobalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=5000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            concurrent_coroutines=1,
            response_type=response_type,  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

        result = await search_engine.asearch(query)
        return result.response

def prepare_local_search(input_dir, community_level=2, temperature=0.5):
    LANCEDB_URI = f"{input_dir}/lancedb"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"

    entity_df = pd.read_parquet(join(input_dir, f"{ENTITY_TABLE}.parquet"))
    entity_embedding_df = pd.read_parquet(join(input_dir, f"{ENTITY_EMBEDDING_TABLE}.parquet"))

    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )

    relationship_df = pd.read_parquet(join(input_dir, f"{RELATIONSHIP_TABLE}.parquet"))
    relationships = read_indexer_relationships(relationship_df)

    # covariate_df = pd.read_parquet(join(input_dir, f"{COVARIATE_TABLE}.parquet"))
    # claims = read_indexer_covariates(covariate_df)
    # covariates = {"claims": claims}

    report_df = pd.read_parquet(join(input_dir, f"{COMMUNITY_REPORT_TABLE}.parquet"))
    reports = read_indexer_reports(report_df, entity_df, community_level)

    text_unit_df = pd.read_parquet(join(input_dir, f"{TEXT_UNIT_TABLE}.parquet"))
    text_units = read_indexer_text_units(text_unit_df)

    llm_config = get_llm_config()
    embedding_config = get_embedding_config()

    llm = ChatOpenAI(
        api_key=llm_config["api_key"],
        api_base=llm_config["api_base"],
        model=llm_config["model"],
        api_type=llm_config["api_type"],
        max_retries=llm_config["max_retries"],
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=embedding_config["api_key"],
        api_base=embedding_config["api_base"],
        api_type=embedding_config["api_type"],
        model=embedding_config["model"],
        deployment_name=embedding_config["deployment_name"],
        max_retries=embedding_config["max_retries"],
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 5000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 1500,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": temperature,
    }

    return llm, context_builder, token_encoder, llm_params, local_context_params

async def local_search(query, input_dir, community_level=2, temperature=0.5, response_type="Multiple Paragraphs"):
    (
        llm, 
        context_builder, 
        token_encoder, 
        llm_params, 
        local_context_params
    ) = prepare_local_search(input_dir, community_level, temperature)

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type=response_type,  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    result = await search_engine.asearch(query)
    return result.response

async def local_question_generate(question_history, input_dir, community_level=2, temperature=0.5):
    (
        llm, 
        context_builder, 
        token_encoder, 
        llm_params, 
        local_context_params
    ) = prepare_local_search(input_dir, community_level, temperature)

    question_generator = LocalQuestionGen(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
    )

    # Ensure question_history is a list of strings (not nested lists)
    # If empty, provide a default starting question
    if not question_history:
        question_history = ["What are the main topics in this dataset?"]
    
    # Flatten any nested lists and ensure all items are strings
    flat_history = []
    for item in question_history:
        if isinstance(item, list):
            flat_history.extend([str(x) for x in item])
        else:
            flat_history.append(str(item))
    
    result = await question_generator.agenerate(
        question_history=flat_history, context_data=None, question_count=5
    )
    return result.response


async def chat_graphrag(
        query, 
        history,
        selected_folder,
        query_type,
        temperature,
        preset,
        show_graph=True
    ):
    # Handle both new format ("output" -> output/artifacts) and old format (timestamp -> output/timestamp/artifacts)
    if selected_folder == "output":
        input_dir = join(script_dir, "output", "artifacts")
    else:
        input_dir = join(script_dir, "output", selected_folder, "artifacts")

    community_level = PRESET_MAPPING[preset]["community_level"]
    response_type = PRESET_MAPPING[preset]["response_type"]

    response = None
    query_entities = []
    
    if query == "/generate":
        # Extract user messages from history (messages format: list of dicts with 'role' and 'content')
        question_history = [msg["content"] for msg in history if msg.get("role") == "user"]
        response = await local_question_generate(
            question_history, input_dir, community_level, temperature
        )
    elif query_type == "global":
        response = await global_search(
            query, input_dir, community_level, temperature, response_type
        )
        # Extract key terms from query for graph visualization
        query_entities = [word for word in query.split() if len(word) > 3]
    elif query_type == "local":
        response = await local_search(
            query, input_dir, community_level, temperature, response_type
        )
        query_entities = [word for word in query.split() if len(word) > 3]
    else:
        response = "Sorry, I can't do a search for you right now"

    print(response)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response})
    
    # Generate graph visualization if enabled
    graph_html = ""
    if show_graph and query != "/generate":
        try:
            # Try to extract mentioned entities from response
            entity_df, _, _ = load_graph_data(input_dir)
            response_entities = extract_entities_from_response(response, entity_df)
            all_entities = list(set(query_entities + response_entities))
            
            graph_html = create_graph_html_for_query(
                input_dir, 
                query_entities=all_entities[:10],
                max_nodes=40
            )
        except Exception as e:
            graph_html = f"<div style='padding: 20px; color: #888;'>Graph visualization unavailable: {str(e)}</div>"
    
    return "", history, graph_html

def list_output_folders():
    """List available output folders for GraphRAG queries.
    
    Supports both old format (timestamped folders like 20241201-123456)
    and new format (direct artifacts/ folder in output/).
    """
    output_dir = join(script_dir, "output")
    if not os.path.exists(output_dir):
        return []
    
    # Check for new GraphRAG format (artifacts directly in output/)
    if os.path.exists(join(output_dir, "artifacts")):
        return ["output"]  # Return "output" as the folder choice
    
    # Check for old format (timestamped folders)
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(join(output_dir, f)) and f[0].isdigit()]
    return sorted(folders, reverse=True)

def create_gradio_interface():
    # Sample prompts for developers to try (based on Student Visa & Athlete Recruitment data)
    SAMPLE_PROMPTS = [
        "What are the main eligibility criteria for student visas across different countries?",
        "Compare the visa requirements between USA (F-1) and UK (Tier 4) student visas",
        "What are the top reasons candidates are rejected for student visas?",
        "How do NCAA eligibility requirements relate to academic performance?",
        "What financial requirements exist for different visa types?",
        "/generate",  # Generate follow-up questions
    ]
    
    custom_css = """
    .contain { display: flex; flex-direction: column; }

    #component-0 { height: 100%; }

    #main-container { display: flex; height: 100%; }

    #right-column { height: calc(100vh - 100px); }

    #chatbot { flex-grow: 1; overflow: auto; }
    
    .sample-prompts { margin-top: 10px; }
    .sample-prompts button { margin: 2px; font-size: 12px; }

    """
    with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="VeritasGraph - GraphRAG Demo") as demo:
        gr.Markdown("""
        # 🔍 VeritasGraph - Graph RAG Demo
        **Enterprise-Grade Knowledge Graph RAG with Verifiable Attribution**
        
        📊 **Dataset:** Student Visa & Admission Eligibility + Elite Athlete Recruitment Analytics
        
        Try the sample prompts below or enter your own question!
        """)
        
        with gr.Row(elem_id="main-container"):
            with gr.Column(scale=1, elem_id="left-column"):
                output_folders = list_output_folders()
                output_folder = output_folders[0] if output_folders else "No output found"
                selected_folder = gr.Dropdown(
                    label="Select Output Folder",
                    choices=output_folders if output_folders else ["No output found"],
                    value=output_folder,
                    interactive=True,
                    allow_custom_value=True
                )

                query_type = gr.Radio(
                    ["global", "local"],
                    label="Query Type",
                    value="global",
                    info="Global: community-based search, Local: entity-based search"
                )

                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=float(0.5)
                )

                preset = gr.Radio(
                    ["Default", "Detailed", "Quick", "Bullet", "Comprehensive", "High-Level", "Focused"],
                    label="Preset",
                    value="Default",
                    info="How specified is the query result"
                )
                
                # Graph visualization toggle
                show_graph = gr.Checkbox(
                    label="🔗 Show Graph Visualization",
                    value=True,
                    info="Display interactive knowledge graph after each query"
                )

            with gr.Column(scale=2, elem_id="right-column"):
                with gr.Tabs():
                    with gr.Tab("💬 Chat", id="chat-tab"):
                        chatbot = gr.Chatbot(
                            label="Chat History", 
                            elem_id="chatbot",
                            height=400,
                            value=[{"role": "assistant", "content": """👋 Welcome to **VeritasGraph**!

I can help you explore the **Student Visa & Athlete Recruitment** knowledge graph.

**📊 Dataset includes:**
- 7,773 student visa candidates (USA F-1, UK Tier 4, Schengen)
- 5,432 elite athletes (FIFA, NCAA, UK GBE compliance)

**Try these sample prompts:**
- "What are the eligibility criteria for student visas?"
- "Compare USA vs UK visa requirements"
- "What causes visa rejections?"
- "How does NCAA eligibility work?"

**Tips:**
- Use **Global Search** for high-level summaries across all data
- Use **Local Search** for specific entity queries (e.g., "NCAA", "F-1 visa")
- Type `/generate` to get AI-suggested follow-up questions
- Toggle **Show Graph Visualization** to see the knowledge graph!

What would you like to know?"""}]
                        )
                    
                    with gr.Tab("🔗 Graph Explorer", id="graph-tab"):
                        gr.Markdown("""
                        ### Interactive Knowledge Graph
                        The graph updates automatically after each query, showing entities and relationships used in the response.
                        
                        **Legend:** 
                        - 🔴 **Red nodes** = Query-related entities
                        - 🔵 **Colored nodes** = Communities (groups of related entities)
                        - **Node size** = Importance (connection count)
                        - **Hover** for entity details | **Drag** to rearrange | **Scroll** to zoom
                        """)
                        graph_display = gr.HTML(
                            value="<div style='padding: 40px; text-align: center; color: #888; background: #0a0a0a; border-radius: 8px; min-height: 500px;'><h3>🔗 Knowledge Graph</h3><p>Run a query to see the related subgraph visualization</p></div>",
                            elem_id="graph-display"
                        )
                
                # Sample prompts as clickable examples
                gr.Markdown("**📝 Sample Prompts (click to use):**", elem_classes=["sample-prompts"])
                with gr.Row():
                    example_btns = []
                    example_prompts = [
                        ("🎓 Visa Criteria", "What are the main eligibility criteria for student visas across different countries?"),
                        ("🆚 USA vs UK", "Compare the visa requirements between USA F-1 and UK Tier 4 student visas"),
                        ("⚽ NCAA Rules", "How do NCAA eligibility requirements relate to academic and athletic performance?"),
                        ("💡 Generate", "/generate"),
                    ]
                
                with gr.Row():
                    for label, prompt in example_prompts[:2]:
                        btn = gr.Button(label, size="sm", variant="secondary")
                        example_btns.append((btn, prompt))
                with gr.Row():
                    for label, prompt in example_prompts[2:]:
                        btn = gr.Button(label, size="sm", variant="secondary")
                        example_btns.append((btn, prompt))
                        
                with gr.Row():
                    query = gr.Textbox(
                        label="Input",
                        placeholder="Enter your query here or click a sample prompt above...",
                        elem_id="query-input",
                        scale=3
                    )
                    query_btn = gr.Button("Send Query", variant="primary")
        
        # Connect example buttons to fill in the query
        for btn, prompt in example_btns:
            btn.click(lambda p=prompt: p, outputs=[query])

        # Query submission with graph visualization
        query.submit(
            fn=chat_graphrag, 
            inputs=[
                query, 
                chatbot,
                selected_folder,
                query_type,
                temperature,
                preset,
                show_graph
            ], 
            outputs=[query, chatbot, graph_display]
        )
        query_btn.click(
            fn=chat_graphrag, 
            inputs=[
                query, 
                chatbot,
                selected_folder,
                query_type,
                temperature,
                preset,
                show_graph
            ], 
            outputs=[query, chatbot, graph_display]
        )
        
        # Standalone graph explorer function
        def explore_full_graph(selected_folder, max_nodes=50):
            """Show the full knowledge graph (top nodes by connectivity)."""
            if selected_folder == "output":
                input_dir = join("output", "artifacts")
            else:
                input_dir = join("output", selected_folder, "artifacts")
            
            return create_graph_html_for_query(input_dir, query_entities=[], max_nodes=max_nodes)
        
        # Add a button to explore full graph
        with gr.Row():
            explore_btn = gr.Button("🔍 Explore Full Graph", variant="secondary", size="sm")
            explore_btn.click(
                fn=explore_full_graph,
                inputs=[selected_folder],
                outputs=[graph_display]
            )

    return demo.queue()


demo = create_gradio_interface()
app = demo.app

# Path to graph cache directory for file serving
GRAPH_CACHE_DIR = os.path.join(os.path.dirname(__file__), "graph_cache")
os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VeritasGraph - GraphRAG Demo")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7861, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (use 0.0.0.0 for external access)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 VeritasGraph - GraphRAG Demo Server")
    print("="*60)
    if args.share:
        print("📡 Creating public shareable link...")
    print(f"🌐 Local URL: http://{args.host}:{args.port}")
    print("="*60 + "\n")
    
    demo.launch(
        server_port=args.port, 
        server_name=args.host,
        share=args.share,
        allowed_paths=[GRAPH_CACHE_DIR]
    )