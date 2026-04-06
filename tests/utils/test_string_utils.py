"""Unit tests for shared string utilities."""

from chatbot.utils import to_snake_case


class TestToSnakeCase:
    """Tests for snake-case normalization."""

    def test_converts_text_to_snake_case(self) -> None:
        # Arrange
        value = "Compliance & Certifications"

        # Act
        result = to_snake_case(value)

        # Assert
        assert result == "compliance_and_certifications"

    def test_strips_extra_separators(self) -> None:
        # Arrange
        value = "  Terms---Of   Service  "

        # Act
        result = to_snake_case(value)

        # Assert
        assert result == "terms_of_service"
