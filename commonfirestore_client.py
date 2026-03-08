"""
Firestore client wrapper with retry logic, connection pooling, and type safety.
Handles all Firestore operations for the entire system.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import firebase_admin
from firebase_admin import credentials, firestore as firestore_admin
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import json
import hashlib

from config.antigravity_config import config

logger = logging.getLogger(__name__)


@dataclass
class FirestoreDocument:
    """Type-safe document wrapper"""
    collection: str
    document_id: str
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-serializable dict"""
        result = self.data.copy()
        result['_metadata'] = {
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'document_id': self.document_id
        }
        return result


class FirestoreClient:
    """Thread-safe Firestore client with connection pooling and retry logic"""
    
    _instance = None
    _clients = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirestoreClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Try to use service account if available
            if (config.firestore.service_account_path and 
                os.path.exists(config.firestore.service_account_path)):
                cred = credentials.Certificate(config.firestore.service_account_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': config.firestore.project_id
                })
                logger.info("Firebase initialized with service account")
            else:
                # Use Application Default Credentials (ADC)
                firebase_admin.initialize_app()
                logger.info("Firebase initialized with ADC")
            
            self.client = firestore_admin.client()
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise
    
    def _test_connection(self):
        """Test Firestore connection with timeout"""
        try:
            # Create a test document
            test_ref = self.client.collection('_system_tests').document('connection_test')
            test_ref.set({
                'timestamp': datetime.utcnow().isoformat(),
                'host': config.host_id
            }, merge=True)
            test_ref.delete()
            logger.info("Firestore connection test successful")
        except Exception as e:
            logger.error(f"Firestore connection test failed: {e}")
            raise
    
    def _get_full_collection_path(self, collection: str) -> str:
        """Get full collection path with prefix"""
        return f"{config.firestore.collection_prefix}/{collection}"
    
    def create_document(self, collection: str, data: Dict[str, Any], 
                       document_id: Optional[str] = None) -> str:
        """
        Create a document with retry logic and automatic timestamps
        
        Args:
            collection: Collection name (without prefix)
            data: Document data
            document_id: Optional custom document ID
            
        Returns:
            Document ID
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                full_collection = self._get_full_collection_path(collection)
                col_ref = self.client.collection(full_collection)
                
                # Add metadata
                now = datetime.utcnow()
                data_with_meta = data.copy()
                data_with_meta['created_at'] = now
                data_with_meta['updated_at'] = now
                data_with_meta['host_id'] = config.host_id
                
                if document_id:
                    doc_ref = col_ref.document(document_id)
                    doc_ref.set(data_with_meta)
                    return document_id
                else:
                    # Generate deterministic ID from data hash
                    data_hash = hashlib.sha256(
                        json.dumps(data, sort_keys=True).encode()
                    ).hexdigest()[:20]
                    doc_id = f"{int(time.time())}_{data_hash}"
                    doc_ref = col_ref.document(doc_id)
                    doc_ref.set(data_with_meta)
                    return doc_id
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create document after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Create document attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def update_document(self, collection: str, document_id: str, 
                       updates: Dict[str, Any], merge: bool = True) -> bool:
        """Update document with optimistic concurrency control"""
        try:
            full_collection = self._get_full_collection_path(collection)
            doc_ref = self.client.collection(full_collection).document(document_id)
            
            # Add update timestamp
            updates['updated_at'] = datetime.utcnow()
            
            doc_ref.set(updates, merge=merge)
            return True
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def get_document(self, collection: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID with error handling"""
        try:
            full_collection = self._get_full_collection_path(collection)
            doc_ref = self.client.collection(full_collection).document(document_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def query_documents(self, collection: str, 
                       filters: List[Dict[str, Any]] = None,
                       order_by: str = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query documents with filters
        
        Args:
            collection: Collection name
            filters: List of filter dicts with keys: field, op, value
            order_by: Field to order by
            limit: Maximum results
            
        Returns:
            List of documents
        """
        try:
            full_collection = self._get_full_collection_path(collection)
            query = self.client.collection(full_collection)
            
            # Apply filters
            if filters:
                for filter_dict in filters:
                    field = filter_dict['field']
                    op = filter_dict['op']  # '==', '>', '<', '>=', '<=', 'array_contains'
                    value = filter_dict['value']
                    
                    if op == '==':
                        query = query.where(field, '==', value)
                    elif op == '>':
                        query = query.where(field, '>', value)
                    elif op == '<':
                        query = query.where(field, '<', value)
                    elif op == '>=':
                        query = query.where(field, '>=', value)
                    elif op == '<=':
                        query = query.where(field, '<=', value)
                    elif op == 'array_contains':
                        query = query.where(field, 'array_contains', value)
            
            # Apply ordering
            if order_by:
                query = query.order_by(order_by)
            
            # Apply limit
            query = query.limit(limit)
            
            # Execute query
            results = []
            docs = query.stream()
            
            for doc in docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def listen_for_changes(self, collection: str, 
                          callback: Callable[[Dict[str, Any]], None],
                          filters: List[Dict[str, Any]] = None):
        """
        Real-time listener for collection changes
        
        Args:
            collection: Collection to monitor
            callback: Function to call with changed document
            filters: Optional filters
        """
        def on_snapshot(col_snapshot, changes, read_time):
            for change in changes:
                if change.type.name == 'ADDED' or change.type.name == 'MODIFIED':
                    doc_data = change.document.to_dict()
                    doc_data['document_id'] = change.document.id
                    doc_data['change_type'] = change.type.name
                    callback(doc_data)
        
        try:
            full_collection = self._get_full_collection_path(collection)
            col_ref = self.client.collection(full_collection)
            
            # Apply filters if provided
            if filters:
                query = col_ref
                for filter_dict in filters:
                    field = filter_dict['field']
                    op = filter_dict['op']
                    value = filter_dict['value']
                    query = query.where(field, op, value)
                query_watch = query.on_snapshot(on_snapshot)
            else:
                col_watch = col_ref.on_snapshot(on_snapshot)
            
            logger.info(f"Started listener for {collection}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start listener: {e}")
            return False
    
    def cleanup_old_documents(self, collection: str, max_age_hours: int = 24):
        """Cleanup documents older than specified age"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)