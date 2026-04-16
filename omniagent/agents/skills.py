"""Skills system for OmniAgent.

Two-root skill discovery:
- <work_dir>/.omniagent/skills/  — project-level (auto-compiled + manual)
- ~/.omniagent/skills/           — user-global
- Formats skills as XML for system prompt injection
- Supports YAML frontmatter stripping and description extraction
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


@dataclass
class Skill:
    """A discovered skill."""

    name: str
    path: Path  # Path to SKILL.md
    root_dir: Path  # Root directory where skill was found
    description: str
    skill_type: str = "prompt"  # "prompt" or "script"


class SkillManager:
    """Manages skill discovery and prompt formatting.

    Two discovery roots (reduced from 3 to minimize lookup overhead):
    1. <work_dir>/.omniagent/skills/  — project-level (auto-compiled + manual)
    2. ~/.omniagent/skills/           — user-global

    Each SKILL.md file represents one skill. Parent directory name = skill name.
    """

    def __init__(
        self,
        work_dir: str | Path,
        max_skills: int = 50,
        max_prompt_chars: int = 30000,
        max_candidates_per_root: int = 300,
    ):
        self.work_dir = Path(work_dir)
        self.max_skills = max_skills
        self.max_prompt_chars = max_prompt_chars
        self.max_candidates_per_root = max_candidates_per_root

        # Discovery roots (priority: high to low)
        self._roots = self._build_roots()

        # Cache
        self._cache: Optional[List[Skill]] = None

    def _build_roots(self) -> List[Path]:
        """Build skill discovery root directories."""
        roots = []

        # 1. <work_dir>/.omniagent/skills/ (project-level: auto-compiled + manual)
        p = self.work_dir / ".omniagent" / "skills"
        if p.is_dir():
            roots.append(p)

        # 2. ~/.omniagent/skills/ (user-global)
        home_skills = Path.home() / ".omniagent" / "skills"
        if home_skills.is_dir():
            roots.append(home_skills)

        return roots

    def _extract_description(self, content: str) -> str:
        """Extract skill description from SKILL.md content.

        Skips YAML frontmatter (between --- delimiters).
        Takes first non-empty, non-heading paragraph as description.
        """
        lines = content.split("\n")
        stripped = []

        # Skip YAML frontmatter
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter:
                stripped.append(line)

        # Find first paragraph
        description_lines = []
        for line in stripped:
            stripped_line = line.strip()
            if not stripped_line:
                if description_lines:
                    break  # End of first paragraph
                continue
            if stripped_line.startswith("#"):
                continue  # Skip headings
            description_lines.append(stripped_line)

        desc = " ".join(description_lines)
        # Truncate long descriptions
        if len(desc) > 200:
            desc = desc[:197] + "..."
        return desc

    @staticmethod
    def _detect_type(skill_dir: Path) -> str:
        """Detect skill type from directory structure.

        Returns 'script' if scripts/main.sh exists, otherwise 'prompt'.
        """
        if (skill_dir / "scripts" / "main.sh").is_file():
            return "script"
        return "prompt"

    def discover_skills(self) -> List[Skill]:
        """Discover all skills from configured roots.

        Returns skills sorted by name. Uses cache if available.
        """
        if self._cache is not None:
            return self._cache

        seen_names = set()
        skills = []

        for root in self._roots:
            count = 0
            # Look for SKILL.md files in subdirectories
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                count += 1
                if count > self.max_candidates_per_root:
                    break

                skill_md = child / "SKILL.md"
                if not skill_md.is_file():
                    continue

                name = child.name
                if name in seen_names:
                    continue  # Higher priority root already has this skill

                try:
                    content = skill_md.read_text(encoding="utf-8")
                    description = self._extract_description(content)
                    skill_type = self._detect_type(child)

                    if not description:
                        description = f"Skill: {name}"

                    skills.append(Skill(
                        name=name,
                        path=skill_md,
                        root_dir=root,
                        description=description,
                        skill_type=skill_type,
                    ))
                    seen_names.add(name)
                except Exception as e:
                    logger.warning(
                        "skill_read_failed",
                        path=str(skill_md),
                        error=str(e),
                    )

        skills.sort(key=lambda s: s.name)
        self._cache = skills

        logger.info(
            "skills_discovered",
            count=len(skills),
            roots=len(self._roots),
        )
        return skills

    def invalidate_cache(self) -> None:
        """Clear the skill discovery cache."""
        self._cache = None

    def format_skills_for_prompt(self) -> str:
        """Format discovered skills as XML for system prompt.

       
        Respects max_skills and max_prompt_chars limits.
        """
        skills = self.discover_skills()

        if not skills:
            return ""

        parts = ["<available-skills>"]
        total_chars = len(parts[0]) + len("</available-skills>")

        for skill in skills:
            type_attr = f' type="{skill.skill_type}"' if skill.skill_type == "script" else ""
            entry = f'<skill name="{skill.name}" path="{skill.path}"{type_attr}>{skill.description}</skill>'
            if total_chars + len(entry) > self.max_prompt_chars:
                parts.append(f"<!-- {len(skills) - len(parts) + 1} more skills truncated -->")
                break
            parts.append(entry)
            total_chars += len(entry)

        parts.append("</available-skills>")
        return "\n".join(parts)

    def get_skill_with_patches(
        self, skill, patches_dir: Optional[Path] = None
    ) -> str:
        """Get skill content merged with any evolution patches."""
        content = skill.path.read_text(encoding="utf-8")

        if patches_dir is None:
            patches_dir = skill.root_dir.parent / "skill_patches"

        patch_file = patches_dir / f"{skill.name}.md"
        if patch_file.exists():
            patches_content = patch_file.read_text(encoding="utf-8")
            if len(patches_content) > 2000:
                patches_content = patches_content[:2000] + "\n... (truncated)"
            content += f"\n\n## Known Corrections (auto-learned)\n{patches_content}"

        return content

    def format_skills_summary(self) -> str:
        """Format a compact skill summary for Sentinel complexity estimation.

        Returns one line per skill: "skill-name: description"
        """
        skills = self.discover_skills()
        if not skills:
            return ""
        lines = [f"- {s.name}: {s.description}" for s in skills]
        return "\n".join(lines)
