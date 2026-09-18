"""Tool registry and execution infrastructure for Nido."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nido.logging import get_logger

logger = get_logger("nido.tools")


@dataclass
class ToolDefinition:
    name: str
    func: Callable[..., Dict[str, Any]]
    description: str
    parameters: Dict[str, Any]


class ToolRegistry:
    """Registry for safe, whitelisted desktop assistant tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
        """Decorator to register a tool function."""

        def decorator(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
            tool_name = name or func.__name__
            doc = description or (func.__doc__ or "").strip()

            # Parse function signature into parameter metadata
            sig = inspect.signature(func)
            parameters: Dict[str, Any] = {}
            for p_name, param in sig.parameters.items():
                if p_name in ("self", "cls"):
                    continue
                type_name = "str"
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        type_name = param.annotation.__name__
                    else:
                        type_name = str(param.annotation)

                param_info: Dict[str, Any] = {
                    "type": type_name,
                    "required": param.default == inspect.Parameter.empty,
                }
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                parameters[p_name] = param_info

            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                func=func,
                description=doc,
                parameters=parameters,
            )
            logger.debug(f"Registered tool: {tool_name}")
            return func

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def execute(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Safely execute a registered tool and return structured output."""
        args = arguments or {}
        tool_def = self.get_tool(tool_name)

        if not tool_def:
            logger.error(f"Execution rejected: Unknown tool '{tool_name}'")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not registered.",
                "tool": tool_name,
            }

        logger.info(f"Executing tool '{tool_name}' with arguments: {args}")
        try:
            result = tool_def.func(**args)
            if not isinstance(result, dict):
                result = {"success": True, "data": result}
            result.setdefault("tool", tool_name)
            return result
        except TypeError as te:
            logger.error(f"Invalid arguments for tool '{tool_name}': {te}")
            return {
                "success": False,
                "error": f"Invalid arguments: {te}",
                "tool": tool_name,
            }
        except Exception as e:
            logger.exception(f"Unexpected error executing tool '{tool_name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }
