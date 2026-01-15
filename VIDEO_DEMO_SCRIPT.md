# Video Demo Script: Financial RAG Chatbot
## Duration: 3-5 Minutes

---

## [0:00 - 0:30] INTRODUCTION & HOOK

**[Screen: Show title slide or IDE with project open]**

"Hi everyone! Today I'm going to demo a Hybrid RAG Chatbot that I built to solve a critical problem with traditional RAG systems. This chatbot answers questions about financial data - specifically fund holdings and trades - and it uses a novel hybrid approach to ensure 100% accuracy on aggregation queries."

**[Quick transition to problem statement]**

"Here's the problem: Pure RAG systems fail when you ask 'Which fund performed best?' because they only retrieve the top-k chunks, which might only contain data from 2-3 funds. The other funds are completely missing, leading to incorrect answers."

---

## [0:30 - 1:15] THE SOLUTION - HYBRID ARCHITECTURE

**[Screen: Show architecture diagram from README or draw on screen]**

"My solution uses a Hybrid Architecture with intelligent query routing."

**[Point to diagram as you explain]**

"When a user asks a question, the Query Router analyzes it and routes it to one of two paths:

1. **Aggregation Path** - For questions like 'Which fund performed best?' or 'Compare all funds', it routes to Pandas, which computes statistics for ALL funds directly from the DataFrame. This ensures no fund is left out.

2. **RAG Path** - For specific queries like 'How many holdings does Garfield have?', it uses semantic search with Pinecone vector database to find the most relevant data chunks.

Both paths then feed their context to Google Gemini LLM, which generates a natural language answer."

---

## [1:15 - 2:00] TECH STACK & KEY FEATURES

**[Screen: Show config.py or key code files]**

"Let me quickly walk through the tech stack:

- **Frontend**: Streamlit for the chat interface
- **Vector Database**: Pinecone Serverless stores 203 embedded chunks
- **Embeddings**: sentence-transformers with all-MiniLM-L6-v2 generates 384-dimensional vectors
- **LLM**: Google Gemini 2.5 Flash for fast, free responses
- **Data Processing**: Pandas handles 1,022 holdings rows and 649 trades rows

**Key features:**
- Smart chunking: 500 tokens per chunk with 50-token overlap
- Relevance validation: Only uses chunks with similarity score ≥ 0.3
- Zero hallucination: Responds 'I cannot find the answer' for out-of-scope queries"

---

## [2:00 - 3:30] LIVE DEMO

**[Screen: Terminal - Run the application]**

"Let me show you how it works. First, I'll start the application."

```bash
streamlit run app.py
```

**[Screen: Streamlit interface loads]**

"The app initializes the RAG system - loads the data, connects to Pinecone, and we're ready to go."

### Demo Query 1: Aggregation Query

**[Click or type in chat]**

"Let's ask: 'Which fund performed best based on yearly P&L?'"

**[Show routing indicator: 📊 Aggregation Query]**

"Notice the system routed this to the Aggregation path. The answer shows ALL funds ranked by P&L - Ytum performed best with over $46 million, followed by MNC Investment Fund and Garfield. This is complete data from all 9 funds."

### Demo Query 2: Specific Query

**[Type new question]**

"Now let's ask: 'How many holdings does Garfield have?'"

**[Show routing indicator: 🔍 Specific Query]**

"This time it routed to the RAG path. It searched Pinecone for relevant chunks about Garfield and found that Garfield has 221 holdings. The semantic search ensures we get precise, fund-specific data."

### Demo Query 3: Out-of-Scope Query

**[Type]**

"What happens with out-of-scope questions? Let me ask: 'What is Apple's market cap?'"

**[Show response]**

"Perfect! It responds: 'Sorry, I cannot find the answer in the provided data.' No hallucinations - it only answers based on what it knows."

---

## [3:30 - 4:00] CODE WALKTHROUGH (Quick Highlights)

**[Screen: Open app.py in IDE]**

"Let me show you the core code. Here in [app.py](app.py), we initialize the Hybrid RAG pipeline using cached resources for fast loading."

**[Scroll to key section around line 68-78]**

"We create a HybridRetrievalPipeline that combines the vector database with the Pandas DataFrames, then wrap it with the LLM handler."

**[Quick switch to query_router.py]**

"In [query_router.py](query_router.py), the router uses regex patterns to classify queries. It looks for keywords like 'which fund best', 'compare funds', or 'top funds' to identify aggregation queries."

**[Show retrieval.py briefly]**

"And in [retrieval.py](retrieval.py), the HybridRetrievalPipeline orchestrates the entire flow - routing, retrieving context, and preparing it for the LLM."

---

## [4:00 - 4:30] WHY THIS MATTERS

**[Screen: Return to presentation or key statistics]**

"Why does this hybrid approach matter?

**Traditional RAG Problems:**
- Retrieves only top-10 chunks
- Might miss 6-7 funds completely
- Gives incomplete or wrong answers

**My Hybrid Solution:**
- Aggregation queries use ALL data
- 100% accuracy on fund comparisons
- Specific queries still benefit from semantic search
- Best of both worlds"

---

## [4:30 - 5:00] CLOSING & FUTURE ENHANCEMENTS

**[Screen: README or project overview]**

"This project demonstrates how combining traditional data processing with modern RAG can solve real-world limitations.

**Future enhancements could include:**
- Multi-language support
- Advanced filtering by date ranges
- Visualization with charts
- Export functionality for reports

The complete code is available in this repository with detailed documentation. Thank you for watching!"

**[Show final screen with GitHub repo or project name]**

---

## PRESENTATION TIPS

### Pacing Guide:
- **Section 1 (Intro)**: 30 seconds - Keep it punchy
- **Section 2 (Architecture)**: 45 seconds - Visual is key
- **Section 3 (Tech Stack)**: 45 seconds - Don't get too technical
- **Section 4 (Demo)**: 90 seconds - This is your wow factor
- **Section 5 (Code)**: 30 seconds - Just highlights
- **Section 6 (Why)**: 30 seconds - Impact statement
- **Section 7 (Closing)**: 30 seconds - Call to action

### Speaking Tips:
1. **Energy**: Speak with enthusiasm but not too fast
2. **Clarity**: Enunciate technical terms clearly
3. **Pauses**: Brief pauses after key points
4. **Screen Time**: Show your face periodically, not just screen
5. **Confidence**: You built something impressive - own it!

### Technical Tips:
1. **Pre-record queries**: Have example questions ready to paste
2. **Clean browser**: Close unnecessary tabs
3. **Zoom in**: Make sure text is readable
4. **Dark mode**: Easier on the eyes for video
5. **Practice transitions**: Smooth switching between windows

### What to Highlight:
- ✅ The problem you solved (RAG aggregation issue)
- ✅ Your novel solution (hybrid routing)
- ✅ Live working demo (proof it works)
- ✅ Clean architecture (good engineering)
- ✅ Real metrics (1,022 holdings, 649 trades, 203 chunks)

### What to Avoid:
- ❌ Reading code line by line
- ❌ Dwelling on setup/installation
- ❌ Explaining basic concepts (what is RAG)
- ❌ Apologizing or being self-deprecating
- ❌ Going over 5 minutes (edit ruthlessly)

---

## BACKUP SLIDES (If Q&A Session)

### Q: Why 384 dimensions instead of 768 or 1536?
**A:** "For structured financial data with clear identifiers like fund names and security tickers, 384 dimensions from all-MiniLM-L6-v2 is sufficient. Higher dimensions are better for nuanced text like essays. Plus, it's 2x faster and runs locally for free."

### Q: What if I want to add more data?
**A:** "Simply update the CSV files and re-run `setup_pinecone.py`. It will re-embed and re-upload everything to Pinecone. The modular architecture makes it easy to scale."

### Q: Can this work with other types of data?
**A:** "Absolutely! The hybrid pattern applies to any domain where you have structured data that requires aggregations plus text that benefits from semantic search - healthcare records, customer data, inventory systems, etc."

---

## PRODUCTION CHECKLIST

Before recording:
- [ ] Test the app thoroughly - ensure it starts without errors
- [ ] Clear chat history for fresh demo
- [ ] Prepare 3-4 example queries ahead of time
- [ ] Check audio levels and background noise
- [ ] Good lighting on your face
- [ ] Clean desktop/browser (no embarrassing tabs)
- [ ] Practice run-through at least twice
- [ ] Have water nearby
- [ ] Disable notifications (Slack, email, etc.)
- [ ] Set "Do Not Disturb" mode on your computer

Good luck with your demo!
