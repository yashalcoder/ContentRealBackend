"""
Background job scheduler for processing scheduled posts
Create this as a new file: app/background_jobs.py
"""

import asyncio
import logging
from datetime import datetime
from app.services.schedule import publish_pending_posts

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_scheduled_posts_job():
    """Background task that runs every minute to process scheduled posts"""
    logger.info("Starting background job for scheduled posts")
    
    while True:
        try:
            logger.info(f"Checking for pending posts at {datetime.now()}")
            publish_pending_posts()
            await asyncio.sleep(60)  # Wait 60 seconds before next check
            
        except Exception as e:
            logger.error(f"Error in background job: {e}")
            await asyncio.sleep(60)  # Wait before retrying

def start_background_jobs():
    """Start all background jobs"""
    asyncio.create_task(run_scheduled_posts_job())