import os
import json
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Career Advisor Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = ""
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Global data storage
careers_df = None
colleges_df = None

# Pydantic models
class StudentAnswers(BaseModel):
    student_id: Optional[str] = None
    answers: Dict[str, Any]

class CareerRecommendation(BaseModel):
    stream: str
    degree: str
    career: str
    score: float
    reason: str

class CollegeRecommendation(BaseModel):
    college_name: str
    district: str
    course: str
    cutoff: Optional[str]
    reason: str

class RecommendationResponse(BaseModel):
    student_id: Optional[str]
    timestamp: str
    careers: List[CareerRecommendation]
    colleges: List[CollegeRecommendation]

# Helper functions
def load_data():
    """Load CSV data at startup"""
    global careers_df, colleges_df
    
    try:
        careers_df = pd.read_csv("data/careers.csv")
        colleges_df = pd.read_csv("data/sample_colleges_db.csv")
        logger.info(f"Loaded {len(careers_df)} careers and {len(colleges_df)} colleges")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def process_student_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw answers to structured profile"""
    
    # Extract interest streams from q1
    interest_streams = [answers.get("q1", "")]
    
    # Extract job preference from q2
    job_pref = answers.get("q2", "").replace(" job", "").replace("Government job", "Government")
    
    # Extract math level from q4
    math_mapping = {
        "MathsComfortable_High": 2,
        "MathsComfortable_Medium": 1,
        "MathsComfortable_Low": 0
    }
    math_level = math_mapping.get(answers.get("q4", ""), 1)
    
    # Extract work type from q5
    work_type = answers.get("q5", "").replace("PreferredWorkType_", "")
    
    # Extract languages from q6
    languages = answers.get("q6", [])
    if isinstance(languages, str):
        languages = [languages]
    
    # Extract boolean preferences
    budget_constraint = answers.get("q7", "") == "BudgetConstraint_Yes"
    relocate = answers.get("q8", "") == "WillingToRelocate_Yes"
    
    # Extract skills from q9
    skills = answers.get("q9", [])
    if isinstance(skills, str):
        skills = [skills]
    
    # Extract goal from q10
    goal = answers.get("q10", "").replace("LongTermGoal_", "")
    
    return {
        "interest_streams": interest_streams,
        "job_preference": job_pref,
        "location_preference": answers.get("q3", ""),
        "math_level": math_level,
        "work_type": work_type,
        "languages": languages,
        "budget_constraint": budget_constraint,
        "relocate_willing": relocate,
        "skills": skills,
        "goal": goal
    }

def filter_careers_for_student(student_profile: Dict[str, Any]) -> pd.DataFrame:
    """Filter careers dataset based on student profile"""
    
    filtered_df = careers_df.copy()
    
    # Filter by interest stream
    interest_streams = student_profile.get("interest_streams", [])
    if interest_streams and interest_streams[0]:
        filtered_df = filtered_df[filtered_df['stream'].isin(interest_streams)]
    
    # Filter by job preference
    job_pref = student_profile.get("job_preference", "")
    if job_pref == "Government":
        filtered_df = filtered_df[filtered_df['job_type'].isin(['Government', 'Both'])]
    elif job_pref == "Private":
        filtered_df = filtered_df[filtered_df['job_type'].isin(['Private', 'Both'])]
    
    # Limit to top 30 for Gemini
    return filtered_df.head(30)

def filter_colleges_for_careers(career_suggestions: List[Dict], student_profile: Dict[str, Any]) -> pd.DataFrame:
    """Filter colleges based on career recommendations and student profile"""
    
    filtered_df = colleges_df.copy()
    
    # Extract degree requirements from career suggestions
    degrees_needed = []
    for career in career_suggestions:
        degree = career.get('degree', '')
        if degree:
            degrees_needed.append(degree)
    
    # Filter colleges that offer these degrees
    if degrees_needed:
        mask = filtered_df['courses_offered'].str.contains('|'.join(degrees_needed), na=False, case=False)
        filtered_df = filtered_df[mask]
    
    # Filter by government preference
    job_pref = student_profile.get("job_preference", "")
    if job_pref == "Government":
        filtered_df = filtered_df[filtered_df['type'] == 'Government']
    
    # Filter by budget constraint
    if student_profile.get("budget_constraint", False):
        # Prefer NA fees (free government colleges) or lower fees
        mask = (filtered_df['fees_annual'] == 'NA') | (pd.to_numeric(filtered_df['fees_annual'], errors='coerce') <= 25000)
        filtered_df = filtered_df[mask]
    
    return filtered_df.head(40)

def call_gemini_for_careers(student_profile: Dict, careers_data: pd.DataFrame) -> List[Dict]:
    """Call Gemini to get career recommendations"""
    
    # Prepare careers data for prompt
    careers_list = careers_data.to_dict('records')
    
    system_prompt = "You are a career advisor. Output ONLY valid JSON, no other text or markdown."
    
    user_prompt = f"""
Student Profile: {json.dumps(student_profile)}

Available Careers: {json.dumps(careers_list[:20])}

Task: Recommend top 5 careers matching this student. Consider their interests, skills, job preferences, math level, and work type preference.

Output exactly this format:
{{
  "careers": [
    {{"stream": "Science", "degree": "B.Tech CSE", "career": "Software Developer", "score": 0.85, "reason": "explanation in max 50 words"}}
  ]
}}
"""
    
    try:
        response = model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=2048,
            )
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        logger.info(f"Gemini careers response: {response_text}")
        
        # Remove any markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        return result.get('careers', [])
        
    except Exception as e:
        logger.error(f"Error calling Gemini for careers: {e}")
        # Fallback: return sample careers based on student profile
        stream = student_profile.get("interest_streams", ["Science"])[0]
        return [{
            "stream": stream,
            "degree": f"B.{stream[:3]}",
            "career": f"{stream} Professional",
            "score": 0.7,
            "reason": "Basic recommendation due to API error - please try again"
        }]

def call_gemini_for_colleges(career_suggestions: List[Dict], colleges_data: pd.DataFrame, student_profile: Dict) -> List[Dict]:
    """Call Gemini to get college recommendations"""
    
    # Prepare colleges data for prompt
    colleges_list = colleges_data.to_dict('records')
    
    system_prompt = "You are a college advisor for J&K students. Output ONLY valid JSON, no other text or markdown."
    
    user_prompt = f"""
Career Recommendations: {json.dumps(career_suggestions)}

Available Colleges: {json.dumps(colleges_list[:25])}

Student Preferences: {json.dumps({
    "budget_constraint": student_profile.get("budget_constraint"),
    "relocate_willing": student_profile.get("relocate_willing"),
    "job_preference": student_profile.get("job_preference")
})}

Task: Recommend top 5 colleges that offer courses matching these careers. Consider course availability, college type, location, and fees.

Output exactly this format:
{{
  "colleges": [
    {{"college_name": "SP College", "district": "Shopian", "course": "B.Tech CSE", "cutoff": "75%", "reason": "explanation in max 30 words"}}
  ]
}}
"""
    
    try:
        response = model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=2048,
            )
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        logger.info(f"Gemini colleges response: {response_text}")
        
        # Remove any markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        return result.get('colleges', [])
        
    except Exception as e:
        logger.error(f"Error calling Gemini for colleges: {e}")
        # Fallback: return sample colleges
        available_colleges = colleges_data.head(3).to_dict('records')
        fallback_colleges = []
        for college in available_colleges:
            courses = college.get('courses_offered', '').split(',')
            fallback_colleges.append({
                "college_name": college.get('college_name', 'Unknown College'),
                "district": college.get('district', 'Unknown'),
                "course": courses[0].strip() if courses else 'General',
                "cutoff": college.get('cutoff_percentage', 'NA'),
                "reason": "Fallback recommendation due to API error"
            })
        return fallback_colleges

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_data()

@app.get("/")
async def root():
    return {"message": "Career Advisor Backend is running", "status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(payload: StudentAnswers):
    """Main endpoint for career and college recommendations"""
    
    try:
        logger.info(f"Processing request for student: {payload.student_id}")
        
        # Step 1: Process student answers
        student_profile = process_student_answers(payload.answers)
        logger.info(f"Student profile: {student_profile}")
        
        # Step 2: Get career recommendations
        filtered_careers = filter_careers_for_student(student_profile)
        career_recommendations = call_gemini_for_careers(student_profile, filtered_careers)
        
        # Step 3: Get college recommendations
        filtered_colleges = filter_colleges_for_careers(career_recommendations, student_profile)
        college_recommendations = call_gemini_for_colleges(career_recommendations, filtered_colleges, student_profile)
        
        # Step 4: Prepare response
        response = RecommendationResponse(
            student_id=payload.student_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            careers=[CareerRecommendation(**career) for career in career_recommendations[:5]],
            colleges=[CollegeRecommendation(**college) for college in college_recommendations[:5]]
        )
        
        logger.info(f"Successfully processed request for student: {payload.student_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "careers_loaded": len(careers_df) if careers_df is not None else 0,
        "colleges_loaded": len(colleges_df) if colleges_df is not None else 0,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)