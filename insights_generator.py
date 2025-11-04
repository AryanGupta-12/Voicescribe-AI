import requests
import json
import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class InsightsGenerator:
    def __init__(self, ollama_url: str = None, model: str = "llama3.2"):
        """
        Initialize the Ollama-based insights generator.
        
        Args:
            ollama_url: Base URL for Ollama API (default: http://localhost:11434)
            model: Ollama model to use (default: llama3.2)
        """
        self.ollama_url = ollama_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.model = model
        self.api_endpoint = f"{self.ollama_url}/api/generate"
    
    def analyze_transcript(self, transcript_text: str, previous_insight: Optional[str] = None) -> Dict:
        """
        Two-stage analysis:
        1. Summarize new transcript
        2. If previous_insight exists → merge & update
        """
        try:
        # --- Stage 1: summarize new transcript ---
            summarize_prompt = self._create_summarize_prompt(transcript_text)
            summary_response = self._call_ollama(summarize_prompt)
            new_summary = summary_response.strip()

            # --- Stage 2: incremental merge ---
            if previous_insight:
                merge_prompt = self._create_merge_prompt(previous_insight, new_summary)
                final_response = self._call_ollama(merge_prompt,temperature=0.5)
            else:
                # First meeting
                analysis_prompt = self._create_analysis_prompt(transcript_text)
                final_response = self._call_ollama(analysis_prompt)

            insights = self._parse_insights(final_response)
            return insights

        except Exception as e:
            print(f"Error analyzing transcript: {e}")
            return {"error": str(e), "status": "failed"}

    
    def _create_analysis_prompt(self, transcript: str) -> str:
        """
        Prompt for analyzing the first meeting transcript and generating a full baseline insight.
        """
        prompt = f"""
        You are an expert meeting analyst and technical project summarizer.

        Analyze the following **first meeting transcript** carefully and generate a comprehensive, structured report.

        TRANSCRIPT:
        {transcript}

        Please produce your analysis in the following markdown structure:

        # Meeting Analysis Report

        ## 1. Project Objective
        Explain what the project is about, its goals, and why it exists.

        ## 2. Discussion Summary
        Summarize the main discussion points in a concise, factual way.

        ## 3. Roadmap & Timeline
        List any mentioned milestones, deadlines, or deliverable dates.

        ## 4. Current Status
        Describe what stage the project is in right now (e.g., planning, setup, development start).

        ## 5. Requirements Discussed
        List the functional or technical requirements discussed during the meeting.

        ## 6. Action Items
        Extract tasks and assignments in bullet form. Format as:
        - [Action item] — Assigned to: [Person or Role]

        ## 7. Key Decisions
        List any critical decisions made in the meeting.

        ## 8. Participant Contributions
        For each speaker, summarize their contributions, questions, and decisions.

        ## 9. Questions & Concerns Raised
        List any open questions, challenges, or risks raised during the discussion.

        ## 10. Next Steps
        Outline what the team agreed to do next and upcoming meetings or deliverables.

        ---

        Be factual, structured, and avoid assumptions.
        If some sections were not discussed, write “(Not discussed in this meeting)”.
        """
        return prompt

    def _create_summarize_prompt(self, transcript: str) -> str:
        """
        Stage 1 – create a concise meeting summary of the new transcript.
        Keeps only essential info for later merging.
        """
        prompt = f"""
    You are an expert meeting summarizer. Summarize the following meeting transcript
    in a concise but information-rich way. Focus on progress, updates, new decisions,
    and blockers. Use bullet points where appropriate.

    TRANSCRIPT:
    {transcript}

    Return output in markdown with sections:
    # Summary
    ## Key Points
    ## Decisions
    ## New Tasks
    ## Blockers / Risks
    """
        return prompt

    def _create_diff_prompt(self, previous_insight: str, new_summary: str) -> str:
        """
        Stage 2a – extract differences between previous insight and new meeting summary.
        Focuses on new facts, progress, or changes in status.
        """
        prompt = f"""
    You are a comparison engine that identifies updates between two meeting summaries.

    PREVIOUS INSIGHT:
    {previous_insight}

    NEW MEETING SUMMARY:
    {new_summary}

    List all differences in clear, concise bullet points under these categories:
    - New progress / completed tasks
    - Updated timelines or deadlines
    - New blockers or risks
    - New action items or decisions
    - Removed or outdated information

    If nothing changed in a category, write "(no change)".
    """
        return prompt


    def _create_merge_prompt(self, previous_insight: str, new_summary: str) -> str:
        """
        Improved Stage 2 – Merge previous insight with new summary to create an updated,
        incremental report that highlights only what changed since the last meeting.
        """
        prompt = f"""
        You are an expert meeting summarizer writing an incremental update report.

        You are updating a project report with new meeting information.
        You have:
        1. The **previous report**, which represents the last known state.
        2. The **new meeting summary**, which includes new progress, changes, or updates.

        Your goals:
        - Focus ONLY on new information, progress, or changes.
        - Do NOT repeat unchanged text or re-explain the project objective.
        - If nothing changed in a section, write “(no update)”.
        - Use concise bullet points and avoid filler words.
        - End with a **CHANGES SINCE LAST MEETING** section summarizing all deltas.

        Structure the report as follows:

        # Meeting Analysis Report
        ## 1. Project Objective
        ## 2. Key Updates Since Last Meeting
        ## 3. Current Status
        ## 4. New Decisions
        ## 5. New Action Items
        ## 6. New Risks or Blockers
        ## 7. Next Steps
        ## CHANGES SINCE LAST MEETING

        Data for reference:
        --- PREVIOUS REPORT ---
        {previous_insight}

        --- NEW MEETING SUMMARY ---
        {new_summary}

        Now produce the updated report using the above structure.
        Keep it factual, concise, and clearly incremental.
        """
        return prompt



    def _call_ollama(self, prompt: str, temperature: float = 0.5) -> str:
        """
        Call the Ollama API to generate insights.
        
        Args:
            prompt: The prompt to send to Ollama
            temperature: Generation temperature (0.0-1.0)
            
        Returns:
            Generated text response
        """
        temperature = max(0.4, min(temperature, 0.7))  # keep between 0.4–0.7
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=300  # 5 minutes timeout for long transcripts
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to Ollama API at {self.ollama_url}: {e}")
    
    def _parse_insights(self, response_text: str) -> Dict:
        """
        Parse the Ollama response into structured insights.
        
        Args:
            response_text: Raw text response from Ollama
            
        Returns:
            Structured dictionary of insights
        """
        # Return the full markdown response along with structured data
        return {
            "status": "success",
            "full_analysis": response_text,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_insights(self, insights: Dict, output_path: str) -> None:
        """
        Save insights to a file.
        
        Args:
            insights: Dictionary containing insights
            output_path: Path to save the insights file
        """
        # Save as markdown if it's a successful analysis
        if insights.get("status") == "success":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(insights["full_analysis"])
        else:
            # Save as JSON if there's an error
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(insights, f, indent=2)


def generate_insights_from_file(transcript_path: str, output_path: str,
                                 ollama_url: str = None, model: str = "llama3.2") -> Dict:
    """
    Generate insights from a transcript file. If a previous insight exists in the same
    directory, perform an incremental update.
    """
    # Read transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    # Detect previous insight
    previous_insight_text = None
    insight_dir = os.path.dirname(output_path)
    existing_insights = [f for f in os.listdir(insight_dir) if f.endswith(".md") and "latest" not in f]
    if existing_insights:
        latest = sorted(existing_insights)[-1]
        latest_path = os.path.join(insight_dir, latest)
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                previous_insight_text = f.read()
        except Exception:
            previous_insight_text = None

    # Generate insights
    generator = InsightsGenerator(ollama_url=ollama_url, model=model)
    insights = generator.analyze_transcript(transcript_text, previous_insight_text)

    # Save insights
    generator.save_insights(insights, output_path)

    return insights
