"""Thread formatting for LLM prompts (Spec Section 11).

Converts Thread objects into flat chronological text with:
- [TARGET] markers on target user posts
- [REPLY to X] tags showing reply targets
- Priority-based truncation to stay within token budget
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from erisk_task2.models import Comment, Thread

logger = logging.getLogger(__name__)

# Approximate tokens per character (conservative)
CHARS_PER_TOKEN = 4


@dataclass
class FormatStats:
    posts_kept_full: int = 0
    posts_truncated: int = 0
    posts_omitted: int = 0
    total_chars: int = 0


@dataclass
class _Node:
    """Internal node in the reply tree."""
    id: str
    author: str
    body: str
    created_utc: str
    is_target: bool
    is_submission: bool = False
    parent_id: str = ""
    children: list[_Node] = field(default_factory=list)


def format_thread(thread: Thread, max_tokens: int = 2000, truncate_chars: int = 100) -> tuple[str, FormatStats]:
    """Format a Thread for LLM prompt insertion.

    Returns (formatted_string, stats).
    """
    stats = FormatStats()
    token_budget = max_tokens * CHARS_PER_TOKEN  # work in chars

    # Handle empty target threads
    if not thread.has_target_text:
        text = (
            f"=== THREAD (Round {thread.round_number}) ===\n"
            f"Title: {thread.title or '[no title]'}\n\n"
            f"[TARGET did not contribute text in this thread]\n"
            f"=== END THREAD ==="
        )
        stats.total_chars = len(text)
        return text, stats

    # Build reply tree
    nodes, id_to_node = _build_tree(thread)

    # Identify target-related node IDs
    target_ids = set()
    if thread.target_is_author:
        target_ids.add(thread.submission_id)
    for c in thread.target_comments:
        target_ids.add(c.comment_id)

    # Mark branches containing target
    target_branch_ids = _find_target_branches(nodes, target_ids)

    # Classify nodes by priority
    p1_nodes: list[_Node] = []  # target posts + direct replies to target
    p2_nodes: list[_Node] = []  # other posts in target branches
    p3_nodes: list[_Node] = []  # non-target posts in target branches (lower priority)
    p4_nodes: list[_Node] = []  # everything else

    for node in _flatten_chronological(nodes):
        if node.is_submission and not node.is_target and node.id not in target_branch_ids:
            # Submission not in target branch — still include title
            continue

        if node.is_target or node.parent_id in target_ids:
            p1_nodes.append(node)
        elif node.id in target_branch_ids:
            p2_nodes.append(node)
        elif node.parent_id in target_branch_ids:
            p3_nodes.append(node)
        else:
            p4_nodes.append(node)

    # Build output with priority-based truncation
    header = (
        f"=== THREAD (Round {thread.round_number}) ===\n"
        f"Title: {thread.title or '[no title]'}\n\n"
    )
    footer = "\n=== END THREAD ==="
    budget = token_budget - len(header) - len(footer)

    lines: list[str] = []
    used = 0

    # Priority 1: always full text
    for node in p1_nodes:
        line = _format_node(node, thread.target_subject, id_to_node)
        if used + len(line) > budget:
            # If even P1 overflows, truncate oldest target posts
            remaining = budget - used
            if remaining > 50:
                line = line[:remaining] + "..."
                lines.append(line)
                stats.posts_truncated += 1
            break
        lines.append(line)
        used += len(line)
        stats.posts_kept_full += 1

    # Priority 2: full text if budget allows
    for node in p2_nodes:
        line = _format_node(node, thread.target_subject, id_to_node)
        if used + len(line) > budget:
            stats.posts_omitted += len(p2_nodes) - p2_nodes.index(node)
            break
        lines.append(line)
        used += len(line)
        stats.posts_kept_full += 1

    # Priority 3: truncated
    for node in p3_nodes:
        line = _format_node(node, thread.target_subject, id_to_node)
        if used + truncate_chars + 30 > budget:
            stats.posts_omitted += len(p3_nodes) - p3_nodes.index(node)
            break
        if len(line) > truncate_chars:
            line = line[:truncate_chars] + f"... [+{len(line) - truncate_chars} more chars]"
            stats.posts_truncated += 1
        else:
            stats.posts_kept_full += 1
        lines.append(line)
        used += len(line)

    # Priority 4: omitted
    stats.posts_omitted += len(p4_nodes)

    result = header + "\n".join(lines) + footer
    stats.total_chars = len(result)
    return result, stats


def _build_tree(thread: Thread) -> tuple[list[_Node], dict[str, _Node]]:
    """Build reply tree from Thread. Returns (root_nodes, id_to_node)."""
    id_to_node: dict[str, _Node] = {}

    # Submission node
    sub_node = _Node(
        id=thread.submission_id,
        author=thread.author,
        body=thread.body,
        created_utc=thread.created_utc,
        is_target=thread.target_is_author,
        is_submission=True,
    )
    id_to_node[thread.submission_id] = sub_node

    # Comment nodes
    for c in thread.comments:
        node = _Node(
            id=c.comment_id,
            author=c.author,
            body=c.body,
            created_utc=c.created_utc,
            is_target=c.is_target,
            parent_id=c.parent_id,
        )
        id_to_node[c.comment_id] = node

    # Link children to parents
    roots = [sub_node]
    for node in id_to_node.values():
        if node.is_submission:
            continue
        parent = id_to_node.get(node.parent_id)
        if parent:
            parent.children.append(node)
        else:
            # Orphan comment (parent not in this thread) — attach to submission
            sub_node.children.append(node)

    return roots, id_to_node


def _find_target_branches(roots: list[_Node], target_ids: set[str]) -> set[str]:
    """Find all node IDs that are in branches containing target user posts."""
    branch_ids: set[str] = set()

    def _mark_ancestors(node: _Node, path: list[str]) -> bool:
        """DFS: returns True if this subtree contains a target node."""
        current_path = path + [node.id]
        has_target = node.id in target_ids or node.is_target

        for child in node.children:
            if _mark_ancestors(child, current_path):
                has_target = True

        if has_target:
            branch_ids.update(current_path)

        return has_target

    for root in roots:
        _mark_ancestors(root, [])

    return branch_ids


def _flatten_chronological(roots: list[_Node]) -> list[_Node]:
    """Flatten tree to chronological list."""
    all_nodes: list[_Node] = []

    def _collect(node: _Node):
        all_nodes.append(node)
        for child in sorted(node.children, key=lambda n: n.created_utc):
            _collect(child)

    for root in roots:
        _collect(root)

    # Sort by timestamp
    all_nodes.sort(key=lambda n: n.created_utc)
    return all_nodes


def _format_node(node: _Node, target_subject: str, id_to_node: dict[str, _Node]) -> str:
    """Format a single node as a line of text."""
    target_tag = " [TARGET]" if node.is_target else ""
    author = node.author

    if node.is_submission:
        prefix = f"[POST] {author}{target_tag}:"
    else:
        parent = id_to_node.get(node.parent_id)
        if parent:
            parent_name = parent.author
            parent_target = " [TARGET]" if parent.is_target else ""
            if parent.is_submission:
                prefix = f"[REPLY to POST by {parent_name}{parent_target}] {author}{target_tag}:"
            else:
                prefix = f"[REPLY to {parent_name}{parent_target}] {author}{target_tag}:"
        else:
            prefix = f"[COMMENT] {author}{target_tag}:"

    body = node.body if node.body else "[no text]"
    return f"{prefix} {body}\n"
