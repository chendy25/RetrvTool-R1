"""
Improved Tool Server - Cleaner, more robust async tool execution server
"""
import asyncio
import inspect
import logging
import time
import weakref
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Any, Set, Union

import fire
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import concurrent.futures

from .utils import hash_requests
from .tools import get_tool_cls, ALL_TOOLS, set_use_tqdm

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEBUG=False

# === MODELS ===
class ActionRequest(BaseModel):
    """Model for incoming action requests with validation"""
    trajectory_ids: List[str] = Field(..., min_items=1)
    actions: List[str] = Field(..., min_items=1)
    extra_fields: Optional[List[Dict[str, Any]]] = None
    finish: Optional[List[bool]] = None
    is_last_step: Optional[List[bool]] = None

    @validator('actions')
    def validate_actions_length(cls, v, values):
        if 'trajectory_ids' in values and len(v) != len(values['trajectory_ids']):
            raise ValueError("Length of actions must match trajectory_ids")
        return v

    @validator('extra_fields')
    def validate_extra_fields_length(cls, v, values):
        if v is not None and 'trajectory_ids' in values and len(v) != len(values['trajectory_ids']):
            raise ValueError("Length of extra_fields must match trajectory_ids")
        return v


class AgentResponse(BaseModel):
    """Model for outgoing agent responses"""
    observations: List[Union[str, dict]]
    dones: List[bool]
    valids: List[bool]
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    concurrent_requests: int
    thread_pool_size: int
    active_tasks: int
    max_concurrent: int
    tools: List[str]
    uptime_seconds: float


# === CONFIGURATION ===
class ServerConfig:
    """Central configuration for server settings"""
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        workers_per_tool: int = 32,
        max_concurrent_requests: int = 64,
        request_timeout: float = None,
        thread_pool_size: Optional[int] = None,
        enable_hashing: bool = True,
        log_level: str = "info"
    ):
        self.host = host
        self.port = port
        self.workers_per_tool = workers_per_tool
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.enable_hashing = enable_hashing
        self.log_level = log_level
        
        # Auto-configure thread pool size based on concurrency needs
        if thread_pool_size is None:
            self.thread_pool_size = max(max_concurrent_requests * 4, 512)
        else:
            self.thread_pool_size = thread_pool_size


# === TOOL MANAGEMENT ===
class AsyncToolManager:
    """Manages all tools and their execution with improved error handling"""
    
    def __init__(self, tool_types: Tuple[str], config: ServerConfig, use_tqdm: bool = False, done_if_invalid: bool = False):
        self.tools: Dict[str, Any] = {}
        self.use_tqdm = use_tqdm
        self.done_if_invalid = done_if_invalid
        self.config = config
        
        set_use_tqdm(use_tqdm)
        self._initialize_tools(tool_types)
        self._setup_thread_pool()
        
    def _setup_thread_pool(self):
        """Initialize thread pool with proper configuration"""
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="tool_worker"
        )
        logger.info(f"Thread pool initialized with {self.config.thread_pool_size} workers")
        
    def _initialize_tools(self, tool_types: Tuple[str]) -> None:
        """Initialize tools with better error handling and logging"""
        # Ensure finish tool is last
        if "finish" in tool_types:
            tool_types = tuple(t for t in tool_types if t != "finish") + ("finish",)
            
        logger.info(f"Initializing tools: {tool_types}")
        
        initialized_tools = []
        failed_tools = []
        
        for tool_type in tool_types:
            try:
                tool_cls = get_tool_cls(tool_type)
                self.tools[tool_type] = tool_cls(num_workers=self.config.workers_per_tool)
                initialized_tools.append(tool_type)
                logger.info(f"✓ Initialized tool: {tool_type}")
            except Exception as e:
                failed_tools.append((tool_type, str(e)))
                logger.error(f"✗ Failed to initialize tool {tool_type}: {e}")
        
        # Initialize finish tool with proper dependencies
        if "finish" not in failed_tools:
            try:
                finish_tool = get_tool_cls("finish")
                self.tools["finish"] = finish_tool(
                    num_workers=self.config.workers_per_tool, 
                    other_tools=[self.tools[t] for t in initialized_tools if t != "finish"]
                )
                logger.info("✓ Initialized finish tool")
            except Exception as e:
                logger.error(f"✗ Failed to initialize finish tool: {e}")
        
        self._log_tool_status()
        
        if failed_tools:
            logger.warning(f"Some tools failed to initialize: {failed_tools}")
    
    def _log_tool_status(self):
        """Log the status of all available tools"""
        logger.info("Tool Status Summary:")
        for tool in ALL_TOOLS:
            status = "🟢 ACTIVE" if tool in self.tools else "⚪ INACTIVE"
            logger.info(f"  {tool}: {status}")
    
    def get_usage_instructions(self) -> str:
        """Generate usage instructions for available tools"""
        instructions = []
        for tool_type, tool in self.tools.items():
            if tool_type not in ["finish", "base"] and hasattr(tool, 'get_usage_inst'):
                instructions.append(f"• {tool_type}: {tool.get_usage_inst()}")
        
        if not instructions:
            return "No tools available for usage instructions."
            
        return "\n".join([
            "Available tools:",
            *instructions
        ])
    
    def _identify_tool_for_action(self, action: str, extra_field: Dict[str, Any]) -> Optional[str]:
        """Identify appropriate tool for a single action"""
        # Check for explicit finish signal
        if extra_field.get("finish", False):
            return "finish"
            
        # Single tool case
        if len(self.tools) == 1:
            return list(self.tools.keys())[0]
        
        # Try each tool (except special ones) to parse action
        for tool_type, tool in self.tools.items():
            if tool_type in ["finish", "mcp_interface"]:
                continue
                
            try:
                if hasattr(tool, 'parse_action'):
                    _, valid = tool.parse_action(action)
                    if valid:
                        return tool_type
            except Exception as e:
                logger.debug(f"Tool {tool_type} parse error: {e}")
                continue
        
        # Try MCP interface as fallback
        if "mcp_interface" in self.tools:
            try:
                _, valid = self.tools["mcp_interface"].parse_action(action)
                if valid:
                    return "mcp_interface"
            except Exception as e:
                logger.debug(f"MCP interface parse error: {e}")

        return None
    
    async def identify_tool_types_batch(self, actions: List[str], extra_fields: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Efficiently identify tools for batch of actions"""
        def process_batch_chunk(chunk_data):
            chunk_actions, chunk_extra_fields = chunk_data
            return [
                self._identify_tool_for_action(action, extra_field)
                for action, extra_field in zip(chunk_actions, chunk_extra_fields)
            ]
        
        # Process in optimal chunks to balance CPU usage and responsiveness
        chunk_size = min(100, max(10, len(actions) // 4))
        tool_types = []
        
        for i in range(0, len(actions), chunk_size):
            chunk_end = min(i + chunk_size, len(actions))
            chunk_data = (actions[i:chunk_end], extra_fields[i:chunk_end])
            
            chunk_results = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                process_batch_chunk,
                chunk_data
            )
            tool_types.extend(chunk_results)
            
            # Yield control periodically for large batches
            if len(actions) > 500 and i % (chunk_size * 5) == 0:
                await asyncio.sleep(0.001)
        
        return tool_types
    
    async def process_actions(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Union[str, dict]], List[bool], List[bool]]:
        """Process batch of actions with improved error handling and performance"""
        
        start_time = time.time()
        num_actions = len(actions)
        
        # Identify tools for all actions
        tool_types = await self.identify_tool_types_batch(actions, extra_fields)
        
        # Initialize results
        observations = [None] * num_actions
        dones = [False] * num_actions
        valids = [False] * num_actions
        
        # Group actions by tool type for efficient batch processing
        tool_groups = self._group_actions_by_tool(tool_types, trajectory_ids, actions, extra_fields)
        
        # Process each tool group
        tasks = []
        for tool_type, (indices, data) in tool_groups.items():
            if tool_type is None:
                # Handle invalid actions
                self._handle_invalid_actions(indices, observations, dones, valids)
                continue
                
            task = self._create_tool_processing_task(tool_type, data)
            tasks.append((tool_type, indices, task))
        
        # Execute all tool tasks concurrently
        await self._execute_tool_tasks(tasks, observations, dones, valids)
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Processed {num_actions} actions in {processing_time:.1f}ms")
        
        return observations, dones, valids
    
    def _group_actions_by_tool(
        self, 
        tool_types: List[Optional[str]], 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Dict[Optional[str], Tuple[List[int], Tuple]]:
        """Group actions by their assigned tool types"""
        groups = {}
        
        for tool_type in set(tool_types):
            indices = [i for i, t in enumerate(tool_types) if t == tool_type]
            if not indices:
                continue
                
            if tool_type is None:
                groups[tool_type] = (indices, None)
            else:
                tool_data = (
                    [trajectory_ids[i] for i in indices],
                    [actions[i] for i in indices],
                    [extra_fields[i] for i in indices]
                )
                groups[tool_type] = (indices, tool_data)
        
        return groups
    
    def _handle_invalid_actions(
        self, 
        indices: List[int], 
        observations: List[Any], 
        dones: List[bool], 
        valids: List[bool]
    ):
        """Handle actions that couldn't be matched to any tool"""
        usage_instructions = self.get_usage_instructions()
        error_response = {
            "obs": "", 
            "invalid_reason": "No valid tool found for action",
            "available_tools": usage_instructions
        }
        
        for idx in indices:
            observations[idx] = error_response
            valids[idx] = False
            dones[idx] = self.done_if_invalid
    
    def _create_tool_processing_task(self, tool_type: str, data: Tuple):
        """Create appropriate task for tool processing (async vs sync)"""
        tool = self.tools[tool_type]
        trajectory_ids, actions, extra_fields = data
        
        # Check if tool has async method
        if hasattr(tool, "aget_observations") and inspect.iscoroutinefunction(tool.aget_observations):
            return asyncio.create_task(
                tool.aget_observations(trajectory_ids, actions, extra_fields)
            )
        else:
            # Use thread pool for sync methods
            return asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                tool.get_observations,
                trajectory_ids,
                actions,
                extra_fields,
            )
    
    async def _execute_tool_tasks(
        self,
        tasks: List[Tuple[str, List[int], Any]],
        observations: List[Any],
        dones: List[bool],
        valids: List[bool]
    ):
        """Execute tool tasks and collect results with proper error handling"""
        for tool_type, indices, task in tasks:
            try:
                if inspect.isawaitable(task):
                    tool_observations, tool_dones, tool_valids = await task
                else:
                    tool_observations, tool_dones, tool_valids = task
                
                # Assign results to correct positions
                for idx_pos, result_idx in enumerate(indices):
                    observations[result_idx] = tool_observations[idx_pos]
                    dones[result_idx] = tool_dones[idx_pos]
                    valids[result_idx] = tool_valids[idx_pos]
                    
            except Exception as e:
                logger.error(f"Tool {tool_type} processing failed: {e}", exc_info=True)
                
                if DEBUG:
                    raise e
                # Handle failed tool processing gracefully
                error_response = {
                    "obs": "", 
                    "error": f"Tool processing failed: {str(e)}",
                    "tool_type": tool_type
                }
                
                for result_idx in indices:
                    observations[result_idx] = error_response
                    dones[result_idx] = True
                    valids[result_idx] = False
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            logger.info("Thread pool shut down")


# === SERVER ===
class AsyncToolServer:
    """Main server class with improved architecture"""
    
    def __init__(
        self,
        tool_types: Tuple[str],
        config: ServerConfig,
        use_tqdm: bool = False,
        done_if_invalid: bool = False,
        use_ray: bool = False,
    ):
        self.config = config
        self.start_time = time.time()
        self.active_requests = 0
        
        # Initialize tool manager
        if use_ray:
            from .ray_utils import RayToolManager
            self.tool_manager = RayToolManager(tool_types, config, use_tqdm, done_if_invalid)
        else:
            self.tool_manager = AsyncToolManager(tool_types, config, use_tqdm, done_if_invalid)
        
        # Request deduplication (if enabled)
        self.processing_cache = weakref.WeakValueDictionary() if config.enable_hashing else {}
        
        # Create app with lifespan management
        self.app = FastAPI(
            title="Async Tool Server",
            description="High-performance async tool execution server",
            version="2.0.0",
            lifespan=self._lifespan
        )
        
        self._setup_routes()
        self._setup_middleware()
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage app lifespan with proper cleanup"""
        logger.info("Server starting up...")
        # Startup logic here if needed
        yield
        logger.info("Server shutting down...")
        self.tool_manager.cleanup()
    
    def _setup_middleware(self):
        """Setup middleware for monitoring and performance"""
        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def _setup_routes(self):
        """Setup API routes with proper validation and error handling"""
        
        # Concurrency limiter
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def get_semaphore():
            """Dependency to manage concurrency"""
            async with semaphore:
                self.active_requests += 1
                try:
                    yield
                finally:
                    self.active_requests -= 1
        
        @self.app.post("/get_observation", response_model=AgentResponse)
        async def process_observations(
            request_data: ActionRequest,
            _: None = Depends(get_semaphore)
        ):
            """Main endpoint for processing observations"""
            start_time = time.time()
            
            try:
                # Process extra fields
                extra_fields = self._prepare_extra_fields(request_data)
                
                # Check for duplicate processing
                if self.config.enable_hashing:
                    cache_key = hash_requests(request_data.model_dump())
                    cached_result = self.processing_cache.get(cache_key)
                    if cached_result:
                        logger.debug(f"Returning cached result for request")
                        return cached_result
                
                # Process actions with timeout
                observations, dones, valids = await asyncio.wait_for(
                    self.tool_manager.process_actions(
                        request_data.trajectory_ids,
                        request_data.actions,
                        extra_fields
                    ),
                    timeout=self.config.request_timeout
                )
                
                processing_time_ms = (time.time() - start_time) * 1000
                response = AgentResponse(
                    observations=observations,
                    dones=dones,
                    valids=valids,
                    processing_time_ms=processing_time_ms
                )
                
                # Cache successful responses
                if self.config.enable_hashing:
                    self.processing_cache[cache_key] = response
                
                return response
                
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Request processing timeout")
            except Exception as e:
                logger.error(f"Request processing failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Comprehensive health check endpoint"""
            thread_pool_size = getattr(self.tool_manager, 'thread_pool', None)
            if thread_pool_size:
                thread_pool_size = thread_pool_size._max_workers
            else:
                thread_pool_size = 0
                
            return HealthResponse(
                status="healthy",
                concurrent_requests=self.active_requests,
                thread_pool_size=thread_pool_size,
                active_tasks=len(self.processing_cache),
                max_concurrent=self.config.max_concurrent_requests,
                tools=list(self.tool_manager.tools.keys()),
                uptime_seconds=time.time() - self.start_time
            )
        
        @self.app.get("/metrics")
        async def metrics():
            """Detailed metrics endpoint"""
            return {
                "active_requests": self.active_requests,
                "cached_responses": len(self.processing_cache),
                "tools_initialized": len(self.tool_manager.tools),
                "uptime_seconds": time.time() - self.start_time,
                "config": {
                    "max_concurrent": self.config.max_concurrent_requests,
                    "timeout": self.config.request_timeout,
                    "hashing_enabled": self.config.enable_hashing,
                }
            }
    
    def _prepare_extra_fields(self, request_data: ActionRequest) -> List[Dict[str, Any]]:
        """Prepare and validate extra fields from request"""
        if request_data.extra_fields:
            extra_fields = request_data.extra_fields
        else:
            extra_fields = [{} for _ in request_data.trajectory_ids]
        
        # Create empty extra fields, take all other fields except trajectory_ids and actions as extra_fields
        keys = set(request_data.model_dump().keys()) - {"trajectory_ids", "actions", "extra_fields"}
        for key in keys:
            if key not in extra_fields[0] and getattr(request_data, key) is not None:
                for ef, value in zip(extra_fields, getattr(request_data, key)):
                    ef[key] = value
        return extra_fields
    
    def start(self):
        """Start the server with optimal configuration"""
        logger.info(f"🚀 Starting Tool Server")
        logger.info(f"   Host: {self.config.host}:{self.config.port}")
        logger.info(f"   Max Concurrent: {self.config.max_concurrent_requests}")
        logger.info(f"   Thread Pool: {self.config.thread_pool_size}")
        logger.info(f"   Timeout: {self.config.request_timeout}s")
        logger.info(f"   Tools: {list(self.tool_manager.tools.keys())}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level,
            access_log=False,  # Disable for performance
            loop="uvloop" if self._has_uvloop() else "asyncio",
            http="httptools",
            timeout_keep_alive=30,
        )
    
    @staticmethod
    def _has_uvloop():
        """Check if uvloop is available for better performance"""
        try:
            import uvloop
            return True
        except ImportError:
            return False


# === MAIN ENTRY POINT ===
def main(
    tool_type: Union[str, Tuple[str]] = "base",
    host: str = "0.0.0.0",
    port: int = 5000,
    workers_per_tool: int = 32,
    max_concurrent_requests: int = 128,
    request_timeout: float = None,
    thread_pool_size: Optional[int] = None,
    use_tqdm: bool = False,
    log_level: str = "info",
    done_if_invalid: bool = False,
    use_ray: bool = False,
    enable_hashing: bool = True,
):
    """Start the tool server with clean configuration"""
    
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level)
    
    # Parse tool types
    if isinstance(tool_type, str):
        tool_types = tuple(t.strip() for t in tool_type.split(","))
    else:
        tool_types = tool_type
    
    # Create configuration
    config = ServerConfig(
        host=host,
        port=port,
        workers_per_tool=workers_per_tool,
        max_concurrent_requests=max_concurrent_requests,
        request_timeout=request_timeout,
        thread_pool_size=thread_pool_size,
        enable_hashing=enable_hashing,
        log_level=log_level
    )
    
    # Create and start server
    server = AsyncToolServer(
        tool_types=tool_types,
        config=config,
        use_tqdm=use_tqdm,
        done_if_invalid=done_if_invalid,
        use_ray=use_ray,
    )
    
    server.start()


if __name__ == "__main__":
    fire.Fire(main)