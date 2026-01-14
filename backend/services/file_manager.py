"""
FileManager: Centralized file path management for the data pipeline.

This module provides a single source of truth for all file paths used across
the application, ensuring consistent directory structure and file handling.
"""

import os
from pathlib import Path
from typing import Optional


class FileManager:
    """
    Centralized manager for all dataset file paths in the pipeline.
    
    Manages the following directory structure:
    - uploads/          : Raw uploaded files
    - uploads/mapped/   : Schema-mapped files
    - uploads/cleaned/  : Cleaned data files
    - uploads/weighted/ : Weighted data files
    - uploads/reports/  : Generated reports
    """
    
    # Base directory for all file operations
    BASE_DIR = Path(__file__).resolve().parent.parent / "temp_uploads"
    
    # Directory structure
    UPLOADS_DIR = BASE_DIR / "uploads" / "default_user"
    MAPPED_DIR = BASE_DIR / "mapped" / "default_user"
    CLEANED_DIR = BASE_DIR / "cleaned" / "default_user"
    WEIGHTED_DIR = BASE_DIR / "weighted" / "default_user"
    REPORTS_DIR = BASE_DIR / "reports" / "default_user"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        Ensure all required directories exist.
        
        Creates any missing directories in the pipeline structure.
        Called automatically by all get_* methods.
        """
        directories = [
            cls.UPLOADS_DIR,
            cls.MAPPED_DIR,
            cls.CLEANED_DIR,
            cls.WEIGHTED_DIR,
            cls.REPORTS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_uploaded_path(cls, filename: str) -> str:
        """
        Get the path for a raw uploaded file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Absolute path to the uploaded file
        """
        cls.ensure_directories()
        return str(cls.UPLOADS_DIR / filename)
    
    @classmethod
    def get_mapped_path(cls, filename: str) -> str:
        """
        Get the path for a schema-mapped file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Absolute path to the mapped file
        """
        cls.ensure_directories()
        return str(cls.MAPPED_DIR / filename)
    
    @classmethod
    def get_cleaned_path(cls, filename: str) -> str:
        """
        Get the path for a cleaned data file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Absolute path to the cleaned file
        """
        cls.ensure_directories()
        return str(cls.CLEANED_DIR / filename)
    
    @classmethod
    def get_weighted_path(cls, filename: str) -> str:
        """
        Get the path for a weighted data file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Absolute path to the weighted file
        """
        cls.ensure_directories()
        return str(cls.WEIGHTED_DIR / filename)
    
    @classmethod
    def get_report_path(cls, filename: str) -> str:
        """
        Get the path for a generated report file.
        
        Args:
            filename: Name of the report file
            
        Returns:
            Absolute path to the report file
        """
        cls.ensure_directories()
        return str(cls.REPORTS_DIR / filename)
    
    @classmethod
    def get_best_available_file(cls, filename: str) -> str:
        """
        Find and return the best available version of a file.
        
        Searches for the file in the following priority order:
        1. Mapped (most processed)
        2. Cleaned
        3. Weighted
        4. Uploaded (raw)
        
        Args:
            filename: Name of the file to search for
            
        Returns:
            Absolute path to the best available version of the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist in any location
        """
        cls.ensure_directories()
        
        # Define search order: most processed to least processed
        search_paths = [
            (cls.MAPPED_DIR / filename, "mapped"),
            (cls.CLEANED_DIR / filename, "cleaned"),
            (cls.WEIGHTED_DIR / filename, "weighted"),
            (cls.UPLOADS_DIR / filename, "uploaded"),
        ]
        
        # Search for the first existing file
        for file_path, stage in search_paths:
            if file_path.exists() and file_path.is_file():
                return str(file_path)
        
        # If no file found, raise detailed error
        raise FileNotFoundError(
            f"File '{filename}' not found in any stage. "
            f"Searched in: mapped, cleaned, weighted, and uploaded directories."
        )
    
    @classmethod
    def file_exists(cls, filename: str, stage: Optional[str] = None) -> bool:
        """
        Check if a file exists in a specific stage or any stage.
        
        Args:
            filename: Name of the file to check
            stage: Optional stage to check ('uploaded', 'mapped', 'cleaned', 
                   'weighted', 'reports'). If None, checks all stages.
                   
        Returns:
            True if file exists, False otherwise
        """
        cls.ensure_directories()
        
        if stage:
            stage_map = {
                "uploaded": cls.UPLOADS_DIR,
                "mapped": cls.MAPPED_DIR,
                "cleaned": cls.CLEANED_DIR,
                "weighted": cls.WEIGHTED_DIR,
                "reports": cls.REPORTS_DIR,
            }
            
            if stage not in stage_map:
                raise ValueError(
                    f"Invalid stage '{stage}'. "
                    f"Must be one of: {', '.join(stage_map.keys())}"
                )
            
            file_path = stage_map[stage] / filename
            return file_path.exists() and file_path.is_file()
        else:
            # Check all stages
            all_dirs = [
                cls.UPLOADS_DIR,
                cls.MAPPED_DIR,
                cls.CLEANED_DIR,
                cls.WEIGHTED_DIR,
                cls.REPORTS_DIR,
            ]
            
            for directory in all_dirs:
                file_path = directory / filename
                if file_path.exists() and file_path.is_file():
                    return True
            
            return False
    
    @classmethod
    def get_file_stage(cls, filename: str) -> Optional[str]:
        """
        Determine which stage a file is in.
        
        Returns the first stage where the file is found, in priority order.
        
        Args:
            filename: Name of the file to locate
            
        Returns:
            Stage name ('mapped', 'cleaned', 'weighted', 'uploaded', 'reports')
            or None if file not found
        """
        cls.ensure_directories()
        
        stage_map = [
            (cls.MAPPED_DIR / filename, "mapped"),
            (cls.CLEANED_DIR / filename, "cleaned"),
            (cls.WEIGHTED_DIR / filename, "weighted"),
            (cls.UPLOADS_DIR / filename, "uploaded"),
            (cls.REPORTS_DIR / filename, "reports"),
        ]
        
        for file_path, stage in stage_map:
            if file_path.exists() and file_path.is_file():
                return stage
        
        return None
    
    @classmethod
    def list_files(cls, stage: str) -> list[str]:
        """
        List all files in a specific stage directory.
        
        Args:
            stage: Stage to list files from ('uploaded', 'mapped', 'cleaned',
                   'weighted', 'reports')
                   
        Returns:
            List of filenames in the specified stage
            
        Raises:
            ValueError: If invalid stage name is provided
        """
        cls.ensure_directories()
        
        stage_map = {
            "uploaded": cls.UPLOADS_DIR,
            "mapped": cls.MAPPED_DIR,
            "cleaned": cls.CLEANED_DIR,
            "weighted": cls.WEIGHTED_DIR,
            "reports": cls.REPORTS_DIR,
        }
        
        if stage not in stage_map:
            raise ValueError(
                f"Invalid stage '{stage}'. "
                f"Must be one of: {', '.join(stage_map.keys())}"
            )
        
        directory = stage_map[stage]
        
        if not directory.exists():
            return []
        
        # Return only files, not directories
        return [
            f.name for f in directory.iterdir() 
            if f.is_file()
        ]
    
    @classmethod
    def get_all_files(cls) -> dict[str, list[str]]:
        """
        Get all files organized by stage.
        
        Returns:
            Dictionary mapping stage names to lists of filenames
        """
        cls.ensure_directories()
        
        stages = ["uploaded", "mapped", "cleaned", "weighted", "reports"]
        
        return {
            stage: cls.list_files(stage)
            for stage in stages
        }
