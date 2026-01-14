"""
File Manager - Handles file operations and storage management
"""

import os
import shutil
import uuid
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib

class FileManager:
    """
    Manages file storage, retrieval, and organization
    """
    
    def __init__(self, base_storage_path: str = "./storage"):
        """
        Initialize file manager
        
        Args:
            base_storage_path: Base directory for file storage
        """
        self.base_storage_path = Path(base_storage_path)
        self.uploads_dir = self.base_storage_path / "uploads"
        self.processed_dir = self.base_storage_path / "processed"
        self.reports_dir = self.base_storage_path / "reports"
        self.temp_dir = self.base_storage_path / "temp"
        
        # Create directories if they don't exist
        self._ensure_directories_exist()
        
        # In-memory file registry (would be database in production)
        self._file_registry: Dict[str, Dict[str, Any]] = {}
    
    def _ensure_directories_exist(self):
        """Ensure all required directories exist"""
        for directory in [self.uploads_dir, self.processed_dir, self.reports_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(
        self,
        file_content: bytes,
        file_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Save uploaded file to storage
        
        Generates unique filename, saves to disk, and registers in file registry
        
        Args:
            file_content: File content as bytes
            file_name: Original file name
            user_id: User identifier
            
        Returns:
            Dictionary with file information
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Extract file extension
        file_ext = Path(file_name).suffix
        
        # Create user directory if needed
        user_dir = self.uploads_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique file path
        unique_filename = f"{file_id}{file_ext}"
        file_path = user_dir / unique_filename
        
        # Save file to disk
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Calculate file hash for integrity
        file_hash = self.calculate_file_hash(str(file_path))
        
        # Create file info
        file_info = {
            "file_id": file_id,
            "file_name": file_name,
            "file_path": str(file_path),
            "file_size": len(file_content),
            "file_hash": file_hash,
            "user_id": user_id,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        # Register in memory (would be database in production)
        self._file_registry[file_id] = file_info
        
        return file_info
    
    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Load file as Pandas DataFrame safely
        
        Handles CSV and Excel files with error handling
        
        Args:
            file_path: Path to file
            
        Returns:
            Pandas DataFrame
            
        Raises:
            ValueError: If file format is not supported
            Exception: If file cannot be parsed
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                # Try different encodings and delimiters
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin-1')
                    except:
                        df = pd.read_csv(file_path, encoding='cp1252')
                        
            elif file_ext in ['.xlsx', '.xls']:
                # Read Excel file
                df = pd.read_excel(file_path, engine='openpyxl' if file_ext == '.xlsx' else None)
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Basic validation
            if df.empty:
                raise ValueError("File is empty or contains no data")
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load file as DataFrame: {str(e)}")
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get the file path for a given file ID
        
        Args:
            file_id: File identifier
            
        Returns:
            File path or None if not found
        """
        file_info = self._file_registry.get(file_id)
        return file_info["file_path"] if file_info else None
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from storage
        
        Args:
            file_id: File identifier
            
        Returns:
            True if deletion successful
        """
        file_info = self._file_registry.get(file_id)
        
        if not file_info:
            return False
        
        file_path = Path(file_info["file_path"])
        
        try:
            # Delete file from disk
            if file_path.exists():
                file_path.unlink()
            
            # Remove from registry
            del self._file_registry[file_id]
            
            return True
        except Exception:
            return False
    
    def delete_file_by_path(self, file_path: str) -> bool:
        """
        Delete a file by its path
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deletion successful
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                
                # Also remove from registry if present
                for file_id, info in list(self._file_registry.items()):
                    if info["file_path"] == file_path:
                        del self._file_registry[file_id]
                        break
                
                return True
            return False
        except Exception:
            return False
    
    def create_temp_file(
        self,
        content: bytes,
        extension: str = ".tmp"
    ) -> str:
        """
        Create a temporary file
        
        Args:
            content: File content
            extension: File extension
            
        Returns:
            Path to temporary file
        """
        temp_filename = f"{uuid.uuid4()}{extension}"
        temp_path = self.temp_dir / temp_filename
        
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        return str(temp_path)
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files
        
        Deletes files in temp directory older than specified hours
        
        Args:
            max_age_hours: Maximum age in hours for temp files
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        current_time = datetime.now()
        max_age_seconds = max_age_hours * 3600
        
        try:
            for file_path in self.temp_dir.rglob('*'):
                if file_path.is_file():
                    file_age = current_time.timestamp() - file_path.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except Exception:
                            continue
        except Exception:
            pass
        
        return deleted_count
    
    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file
        
        Reads file in chunks to handle large files efficiently
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            return sha256_hash.hexdigest()
        except Exception:
            return "hash_error"
    
    def get_user_storage_usage(self, user_id: str) -> Dict[str, Any]:
        """
        Get storage usage statistics for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Storage usage statistics
        """
        total_bytes = 0
        file_count = 0
        by_type = {}
        
        for file_id, file_info in self._file_registry.items():
            if file_info["user_id"] == user_id:
                file_count += 1
                total_bytes += file_info["file_size"]
                
                # Track by file extension
                file_ext = Path(file_info["file_name"]).suffix
                by_type[file_ext] = by_type.get(file_ext, 0) + 1
        
        return {
            "user_id": user_id,
            "total_bytes": total_bytes,
            "file_count": file_count,
            "by_type": by_type
        }
    
    def archive_file(self, file_id: str) -> bool:
        """
        Archive a file (move to archive storage)
        
        Args:
            file_id: File identifier
            
        Returns:
            True if archival successful
        """
        # TODO: Implement file archival
        return False
    
    def restore_file(self, file_id: str) -> bool:
        """
        Restore an archived file
        
        Args:
            file_id: File identifier
            
        Returns:
            True if restoration successful
        """
        # TODO: Implement file restoration
        return False
    
    def list_user_files(
        self,
        user_id: str,
        file_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all files for a user
        
        Args:
            user_id: User identifier
            file_type: Optional file type filter (e.g., '.csv', '.xlsx')
            
        Returns:
            List of file metadata dictionaries
        """
        user_files = []
        
        for file_id, file_info in self._file_registry.items():
            if file_info["user_id"] == user_id:
                if file_type is None or file_info["file_name"].endswith(file_type):
                    user_files.append(file_info.copy())
        
        # Sort by upload timestamp (newest first)
        user_files.sort(key=lambda x: x["upload_timestamp"], reverse=True)
        
        return user_files
    
    def ensure_directory_exists(self, directory_path: str) -> bool:
        """
        Ensure a directory exists, create if it doesn't
        
        Args:
            directory_path: Path to directory
            
        Returns:
            True if directory exists or was created
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    def save_cleaned_file(self, df, file_id: str):
        cleaned_dir = os.path.join(self.base_storage_path, "cleaned", "default_user")
        os.makedirs(cleaned_dir, exist_ok=True)

        cleaned_path = os.path.join(cleaned_dir, f"{file_id}_cleaned.csv")

        df.to_csv(cleaned_path, index=False)

        return cleaned_path


    def get_cleaned_file_path(self, file_id: str):
        cleaned_path = os.path.join(
            self.base_storage_path,
            "cleaned",
            "default_user",
            f"{file_id}_cleaned.csv"
        )
        return cleaned_path if os.path.exists(cleaned_path) else None
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from registry
        
        Args:
            file_id: File identifier
            
        Returns:
            File info dictionary or None if not found
        """
        return self._file_registry.get(file_id)
    
    def get_cleaned_path(self, file_id: str) -> str:
        """
        Get path to cleaned file
        
        Args:
            file_id: File identifier
            
        Returns:
            Path to cleaned file
            
        Raises:
            FileNotFoundError: If cleaned file does not exist
        """
        cleaned_dir = self.base_storage_path / "cleaned" / "default_user"
        cleaned_file = os.path.join(cleaned_dir, f"{file_id}_cleaned.csv")
        
        if not os.path.exists(cleaned_file):
            raise FileNotFoundError(f"404: File not found: {file_id}")
        
        return cleaned_file
    
    def get_processed_path(self, file_id: str) -> str:
        """
        Get path to processed file
        
        Args:
            file_id: File identifier
            
        Returns:
            Path to processed file
        """
        return str(self.processed_dir / f"{file_id}.csv")
    
    def get_weighted_path(self, file_id: str) -> str:
        """
        Get path to weighted file
        
        Args:
            file_id: File identifier
            
        Returns:
            Path to weighted file
        """
        weighted_dir = self.base_storage_path / "weighted" / "default_user"
        return str(weighted_dir / f"{file_id}_weighted.csv")
    
    def get_upload_path(self, file_id: str) -> str:
        """
        Get path to uploaded file
        
        Args:
            file_id: File identifier
            
        Returns:
            Path to uploaded file
        """
        upload_dir = self.uploads_dir / "default_user"
        return str(upload_dir / f"{file_id}.csv")
    
    def get_best_available_file(self, file_id: str) -> str:
        """
        Get the best available file for a given file_id.
        
        Tries in order:
        1. Cleaned file (preferred)
        2. Mapped file (fallback)
        3. Uploaded file (last fallback)
        
        Args:
            file_id: File identifier
            
        Returns:
            Path to the best available file
            
        Raises:
            FileNotFoundError: If no file is found
        """
        cleaned_dir = self.base_storage_path / "cleaned" / "default_user"
        mapped_dir = self.base_storage_path / "mapped" / "default_user"
        upload_dir = self.uploads_dir / "default_user"
        
        cleaned_path = os.path.join(cleaned_dir, f"{file_id}_cleaned.csv")
        mapped_path = os.path.join(mapped_dir, f"{file_id}_mapped.csv")
        uploaded_path = os.path.join(upload_dir, f"{file_id}.csv")
        
        if os.path.exists(cleaned_path):
            return cleaned_path
        if os.path.exists(mapped_path):
            return mapped_path
        if os.path.exists(uploaded_path):
            return uploaded_path
        
        raise FileNotFoundError(f"404: File not found: {file_id}")

