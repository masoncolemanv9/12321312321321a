"""Tests for the Markdown-ish → Telegram-HTML reformatter."""

from __future__ import annotations

from bot.addons.pretty_text.core import format_for_telegram


def test_empty_input_returns_empty_string() -> None:
    assert format_for_telegram("") == ""
    assert format_for_telegram("   \n  \n") == ""


def test_plain_paragraph_passes_through_with_inline_escaping() -> None:
    out = format_for_telegram("Hello & world < 5")
    assert "&amp;" in out
    assert "&lt;" in out
    # No spurious tags should appear for plain text.
    assert "<b>" not in out
    assert "<i>" not in out


def test_bold_double_star_becomes_b_tag() -> None:
    out = format_for_telegram("This is **bold** text.")
    assert "<b>bold</b>" in out


def test_italic_single_star_becomes_i_tag() -> None:
    out = format_for_telegram("Some *italic* word.")
    assert "<i>italic</i>" in out


def test_bold_and_italic_dont_overlap() -> None:
    out = format_for_telegram("**bold** and *italic*")
    assert "<b>bold</b>" in out
    assert "<i>italic</i>" in out
    # The double-star match must win, so we never see <b><i></i></b>.
    assert "<b><i>" not in out


def test_inline_code_is_html_escaped_inside_tag() -> None:
    out = format_for_telegram("Use `<div>` here.")
    assert "<code>&lt;div&gt;</code>" in out


def test_link_renders_as_anchor() -> None:
    out = format_for_telegram("Visit [Render](https://render.com) docs.")
    assert '<a href="https://render.com">Render</a>' in out


def test_heading_h1_renders_bold_with_marker() -> None:
    out = format_for_telegram("# Заголовок")
    assert "<b>Заголовок</b>" in out
    assert "🔹" in out


def test_heading_h2_renders_bold_with_marker() -> None:
    out = format_for_telegram("## Подзаголовок")
    assert "<b>Подзаголовок</b>" in out
    assert "▸" in out


def test_unordered_list_uses_bullets() -> None:
    out = format_for_telegram("- one\n- two\n- three")
    assert "• one" in out
    assert "• two" in out
    assert "• three" in out


def test_ordered_list_preserves_numbers() -> None:
    out = format_for_telegram("1. first\n2. second\n5. fifth")
    assert "1. first" in out
    assert "2. second" in out
    assert "5. fifth" in out


def test_horizontal_rule_renders_as_box_chars() -> None:
    out = format_for_telegram("Above\n\n---\n\nBelow")
    assert "━" in out


def test_blockquote_wraps_in_blockquote_tag() -> None:
    out = format_for_telegram("> quoted line 1\n> quoted line 2")
    assert "<blockquote>" in out
    assert "</blockquote>" in out
    assert "quoted line 1" in out


def test_fenced_code_block_uses_pre_tag() -> None:
    src = "before\n\n```python\nx = 1\nprint(x)\n```\n\nafter"
    out = format_for_telegram(src)
    assert "<pre>" in out
    assert "x = 1" in out
    assert "print(x)" in out


def test_fenced_code_block_escapes_html_inside() -> None:
    src = "```\n<script>alert(1)</script>\n```"
    out = format_for_telegram(src)
    assert "&lt;script&gt;" in out
    # No literal closing </script> tag may leak out raw.
    assert "</script>" not in out


def test_markdown_table_becomes_pre_with_box_chars() -> None:
    src = (
        "| Сервер | Цена | RAM |\n"
        "|--------|------|-----|\n"
        "| Render | $7   | 512 |\n"
        "| VPS    | €4   | 2GB |\n"
    )
    out = format_for_telegram(src)
    assert "<pre>" in out
    assert "┌" in out and "┐" in out
    assert "│" in out
    assert "Render" in out and "VPS" in out


def test_markdown_table_with_pipe_only_no_separator_also_renders() -> None:
    """A row that *looks* like a table but lacks the |---| separator
    should still parse as paragraph text — we don't force-table it."""
    src = "Option A | Option B"
    out = format_for_telegram(src)
    # No <pre> block from a pseudo-table without separator.
    assert "<pre>" not in out


def test_output_is_truncated_to_telegram_limit() -> None:
    huge = "abcde " * 2000  # ~12 000 chars
    out = format_for_telegram(huge)
    assert len(out) <= 4096
    assert "обрезано" in out


def test_no_unbalanced_tags_in_simple_output() -> None:
    out = format_for_telegram("**bold** *italic* `code` and [link](https://x.com)")
    # Count opens and closes of each tag are equal.
    for tag in ("b", "i", "code", "a"):
        assert out.count(f"<{tag}") == out.count(f"</{tag}>")


def test_real_world_mixed_block() -> None:
    src = (
        "# План\n"
        "\n"
        "Дальше:\n"
        "- купить молоко\n"
        "- испечь хлеб\n"
        "\n"
        "## Бюджет\n"
        "\n"
        "| статья | сумма |\n"
        "|--------|-------|\n"
        "| еда    | 5000  |\n"
        "| свет   | 1500  |\n"
        "\n"
        "> важно: не забыть про **скидку**\n"
    )
    out = format_for_telegram(src)
    assert "<b>План</b>" in out
    assert "• купить молоко" in out
    assert "<pre>" in out
    assert "<blockquote>" in out
    assert "<b>скидку</b>" in out
