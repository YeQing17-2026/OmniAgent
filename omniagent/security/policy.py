"""Tool policy system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from omniagent.infra import get_logger

logger = get_logger(__name__)


class PolicyDecision(str, Enum):
    """Policy decision."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolProfile(str, Enum):
    """Tool profile."""

    MINIMAL = "minimal"  # Only read-only tools
    CODING = "coding"  # File tools + bash (safe commands)
    FULL = "full"  # All tools


# Tool groups
TOOL_GROUPS = {
    "fs": ["read_file", "write_file", "edit_file"],
    "search": ["grep", "find", "ls", "diff"],
    "runtime": ["bash"],
    "web": ["web_search", "web_fetch"],
    "json": ["load_json", "save_json"],
    "process": ["process_list", "process_kill"],
    "memory": ["memory_search", "memory_get"],
}

# Profile configurations
PROFILE_CONFIGS = {
    ToolProfile.MINIMAL: {
        "allowed_tools": ["read_file"],
        "allowed_groups": ["memory", "search"],
        "require_approval": [],
    },
    ToolProfile.CODING: {
        "allowed_tools": ["read_file", "write_file", "edit_file", "bash", "load_json", "save_json", "process_list"],
        "allowed_groups": ["fs", "search", "runtime", "json", "process", "memory", "web"],
        "require_approval": ["bash", "process_kill"],
    },
    ToolProfile.FULL: {
        "allowed_tools": "*",
        "allowed_groups": "*",
        "require_approval": ["bash", "process_kill"],
    },
}


@dataclass
class PolicyRule:
    """Policy rule."""

    name: str
    decision: PolicyDecision
    tools: Optional[List[str]] = None  # Specific tools
    groups: Optional[List[str]] = None  # Tool groups
    priority: int = 0  # Higher priority wins


class ToolPolicy:
    """Tool policy system."""

    def __init__(
        self,
        profile: ToolProfile = ToolProfile.CODING,
        custom_rules: Optional[List[PolicyRule]] = None,
    ):
        """
        Initialize tool policy.

        Args:
            profile: Tool profile (minimal, coding, full)
            custom_rules: Custom policy rules
        """
        self.profile = profile
        self.rules: List[PolicyRule] = []

        # Load profile rules
        self._load_profile(profile)

        # Add custom rules
        if custom_rules:
            self.rules.extend(custom_rules)

        # Sort rules by priority (descending)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

        logger.info(
            "policy_initialized",
            profile=profile.value,
            rule_count=len(self.rules),
        )

    def check_tool(self, tool_name: str) -> PolicyDecision:
        """
        Check if tool is allowed.

        Args:
            tool_name: Tool name

        Returns:
            Policy decision
        """
        # Check rules in priority order
        for rule in self.rules:
            # Check specific tools
            if rule.tools:
                if tool_name in rule.tools:
                    logger.debug(
                        "policy_match",
                        tool=tool_name,
                        rule=rule.name,
                        decision=rule.decision.value,
                    )
                    return rule.decision

            # Check groups
            if rule.groups:
                for group in rule.groups:
                    if group == "*" or (
                        group in TOOL_GROUPS and tool_name in TOOL_GROUPS[group]
                    ):
                        logger.debug(
                            "policy_match",
                            tool=tool_name,
                            rule=rule.name,
                            group=group,
                            decision=rule.decision.value,
                        )
                        return rule.decision

        # Default: deny
        logger.warning("policy_no_match", tool=tool_name, default="deny")
        return PolicyDecision.DENY

    def is_allowed(self, tool_name: str) -> bool:
        """
        Check if tool is allowed (without approval).

        Args:
            tool_name: Tool name

        Returns:
            True if allowed
        """
        decision = self.check_tool(tool_name)
        return decision == PolicyDecision.ALLOW

    def requires_approval(self, tool_name: str) -> bool:
        """
        Check if tool requires approval.

        Args:
            tool_name: Tool name

        Returns:
            True if approval required
        """
        decision = self.check_tool(tool_name)
        return decision == PolicyDecision.REQUIRE_APPROVAL

    def get_allowed_tools(self) -> Set[str]:
        """
        Get all allowed tools.

        Returns:
            Set of allowed tool names
        """
        allowed = set()

        for rule in self.rules:
            if rule.decision == PolicyDecision.ALLOW:
                # Add specific tools
                if rule.tools:
                    if "*" in rule.tools:
                        # All tools - return all known tools
                        for group_tools in TOOL_GROUPS.values():
                            allowed.update(group_tools)
                    else:
                        allowed.update(rule.tools)

                # Add group tools
                if rule.groups:
                    if "*" in rule.groups:
                        # All groups
                        for group_tools in TOOL_GROUPS.values():
                            allowed.update(group_tools)
                    else:
                        for group in rule.groups:
                            if group in TOOL_GROUPS:
                                allowed.update(TOOL_GROUPS[group])

        return allowed

    def add_rule(self, rule: PolicyRule) -> None:
        """
        Add custom rule.

        Args:
            rule: Policy rule
        """
        self.rules.append(rule)
        # Re-sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)

        logger.info("policy_rule_added", rule=rule.name, priority=rule.priority)

    def _load_profile(self, profile: ToolProfile) -> None:
        """Load profile configuration."""
        config = PROFILE_CONFIGS.get(profile, {})

        allowed_tools = config.get("allowed_tools", [])
        allowed_groups = config.get("allowed_groups", [])
        require_approval = config.get("require_approval", [])

        # Create allow rule
        if allowed_tools or allowed_groups:
            self.rules.append(
                PolicyRule(
                    name=f"profile_{profile.value}_allow",
                    decision=PolicyDecision.ALLOW,
                    tools=allowed_tools if allowed_tools != "*" else ["*"],
                    groups=allowed_groups if allowed_groups != "*" else ["*"],
                    priority=10,
                )
            )

        # Create approval rule (higher priority)
        if require_approval:
            self.rules.append(
                PolicyRule(
                    name=f"profile_{profile.value}_approval",
                    decision=PolicyDecision.REQUIRE_APPROVAL,
                    tools=require_approval,
                    priority=20,
                )
            )

        logger.debug(
            "profile_loaded",
            profile=profile.value,
            rules=len(self.rules),
        )
