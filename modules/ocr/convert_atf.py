import re
from collections import Counter
from typing import Optional


class ParsedATF:
    """Represents a parsed ATF document with methods to extract data."""

    # Face types
    ALL_FACES = [
        "obverse",
        "reverse",
        "left",
        "right",
        "top",
        "bottom",
    ]

    def __init__(
        self, transliterations: dict, unicodes: dict, info: dict, used_signs: set
    ):
        """
        Initialize parsed ATF data.

        Args:
            transliterations: Dictionary mapping face names to transliteration line lists
            unicodes: Dictionary mapping face names to unicode line lists
            info: Metadata dictionary (e.g., language)
        """
        self._transliterations = transliterations
        self._unicodes = unicodes
        self._info = info
        self._used_signs = used_signs

    def get_used_signs(self) -> set[str]:
        """Get the set of used signs."""
        return self._used_signs

    def get_transliteration(self, face: str) -> Optional[str]:
        """
        Get the transliteration for a given face.

        Args:
            face: The face name (e.g., 'obverse', 'reverse')

        Returns:
            The transliteration as a string with lines separated by newlines,
            or None if the face has no content
        """
        if face in self._transliterations:
            return self._transliterations[face]
        return None

    def get_unicode(self, face: str) -> Optional[str]:
        """
        Get the unicode representation for a given face.

        Args:
            face: The face name (e.g., 'obverse', 'reverse')

        Returns:
            The unicode representation as a string with lines separated by newlines,
            or None if the face has no content
        """
        if face in self._unicodes:
            return self._unicodes[face]
        return None

    def get_all_unicodes(self) -> dict[str, Optional[str]]:
        """
        Get unicode for all faces.

        Returns:
            Dictionary mapping face names to unicode strings
        """
        return {
            f"{face}_unicode": self.get_unicode(face)
            for face in self.ALL_FACES
            if self.get_unicode(face) is not None
        }

    def get_all_transliterations(self) -> dict[str, Optional[str]]:
        """
        Get transliteration for all faces.

        Returns:
            Dictionary mapping face names to transliteration strings
        """
        return {
            f"{face}_transliteration": self.get_transliteration(face)
            for face in self.ALL_FACES
            if self.get_transliteration(face) is not None
        }

    @property
    def info(self) -> dict:
        """Get parsing info (e.g., language)."""
        return self._info


class ATFConverter:
    """Converter for ATF (ASCII Transliteration Format) cuneiform text."""

    # Face types
    ALL_FACES = [
        "obverse",
        "reverse",
        "left",
        "right",
        "top",
        "bottom",
    ]

    FACE_REMAPPING = {
        "surface a": "obverse",
        "surface b": "reverse",
    }

    # Special tokens
    SPECIAL_TOKENS = [
        "<B>",  # broken
        "<M>",  # missing one or more token?
        "<S>",  # blank space
        "<D>",  # divine
        "<munus>",  # young woman, or woman
        "<ansze>",
        "<ki>",
        "<disz>",
        "x",  # unknown signs
    ]

    def __init__(self, token_path: str = "./data/cuneiform_vocab.tsv"):
        """
        Initialize the ATF converter.

        Args:
            token_path: Path to the cuneiform vocabulary file
        """
        self.text2sign = self._load_token_mapping(token_path)

        # Counters for statistics
        self.vocab_freq = Counter()
        self.new_tokens = Counter()
        self.langs = Counter()
        self.unknown_faces = Counter()

    def _load_token_mapping(self, token_path: str) -> tuple[dict, dict]:
        """Load the text to sign and sign to text mappings."""

        text2sign = {}
        for t in open(token_path).readlines():
            try:
                k, s = t.strip("\n").split("\t")
            except:
                print(t)
                continue
            text2sign[k] = s.replace(" ", "")

        return text2sign

    def _remove_at(self, x: str) -> Optional[str]:
        """Remove @c or @t suffixes from tokens."""
        if x.endswith("@c)") or x.endswith("@t)"):
            return x[:-3] + ")"
        return None

    def _remove_spaces(self, x: list[str]) -> list[str]:
        """Remove consecutive space tokens."""
        new_x = []
        for item in x:
            if item == "<S>" and len(new_x) > 0 and new_x[-1] == "<S>":
                continue
            new_x.append(item)
        return new_x

    def parse(self, raw_text: str) -> Optional[ParsedATF]:
        """
        Parse ATF text and extract transliterations and unicode.

        Args:
            raw_text: The raw ATF text to parse

        Returns:
            ParsedATF object if parsing succeeded, None if the language is not supported
        """
        token_text = {"default": []}
        info = {}

        curr_face = "default"
        sep = "\n"
        if "\\n" in raw_text:
            sep = "\\n"

        for line in raw_text.split(sep):
            line = line.strip()

            if line.startswith("&") or line.startswith("'&"):
                # metadata
                pass
            elif line.startswith("#atf"):
                info["lang"] = line.split("lang ")[-1].strip()
                self.langs[info["lang"]] += 1
                if info["lang"] not in ["sux", "akk", "sux, akk", "akk _sux"]:
                    # do not process those not sux or akk
                    return None
            elif (
                line.startswith("#")
                or line.startswith(">>")
                or line.startswith("<<")
                or line.startswith("||")
            ):
                # comment/link
                continue
            elif line.startswith("$"):
                if "broken" in line:
                    try:
                        token_text[curr_face].append("<B>")
                    except:
                        continue
            elif line.startswith("@"):
                key = line[1:].strip().strip("?")
                if key in self.ALL_FACES:
                    curr_face = key
                    token_text[key] = []
                elif key.startswith("column"):
                    token_text[curr_face].append("<COL>")
                else:
                    self.unknown_faces[key] += 1
            else:
                # Process line content
                self._process_line_content(line, curr_face, token_text)

        # Build transliterations and unicodes from token_text
        transliterations, unicodes, used_signs = self._build_outputs(token_text)
        return ParsedATF(transliterations, unicodes, info, used_signs)

    def _process_line_content(self, line: str, curr_face: str, token_text: dict):
        """Process a content line and extract tokens."""
        # Special symbols
        line = line.replace("{d}", "<D>")

        for x in re.findall(r"\{.*?\}", line):
            line = line.replace(x, " " + x[1:-1] + " ")

        line = line.replace("($ blank space $)", "<S>")

        # Remove underscore
        line = line.replace("_", " ")

        # Remove ending hash #
        line = line.replace("#", "")

        # Remove question mark, exclamation mark
        line = line.replace("?", "")
        line = line.replace("!", "")

        # Remove [] and ()
        for x in re.findall(r"\[.*?\]", line):
            line = line.replace(x, "")

        line = line.split(". ")

        if len(line) >= 2:
            # Make sure only leading line number is split
            if len(line) > 2:
                line = line[0], ". ".join(line[1:])

            line_num, text = line
            if curr_face != "":
                tokens = text.split(" ")
                signs = []
                for i, t in enumerate(tokens):
                    #     if i > 0 and len(signs) > 0:
                    #         signs.append("<S>")  # insert a space between words

                    if "-" in t:
                        ts = t.split("-")
                        for x in ts:
                            x = x.strip()
                            if len(x) == 0:
                                continue
                            if x in self.text2sign:
                                self.vocab_freq[x] += 1
                                signs.append(self.text2sign[x])
                            else:
                                new_x = self._remove_at(x)
                                if new_x and new_x in self.text2sign:
                                    signs.append(self.text2sign[new_x])
                                else:
                                    self.new_tokens[x] += 1
                    elif t in self.text2sign:
                        signs.append(self.text2sign[t])
                    elif t in self.SPECIAL_TOKENS:
                        self.vocab_freq[t] += 1
                        signs.append(t)
                    else:
                        new_x = self._remove_at(t)
                        if new_x and new_x in self.text2sign:
                            signs.append(self.text2sign[new_x])
                        else:
                            if len(t.strip()) > 0:
                                self.new_tokens[t] += 1

                signs = self._remove_spaces(signs)
                token_text[curr_face].append(
                    {"raw": text, "num": line_num, "sign": signs}
                )

    def _build_outputs(
        self, token_text: dict
    ) -> tuple[dict[str, list[list[str]]], dict[str, list[list[str]]], set[str]]:
        """Build transliterations and unicode outputs from parsed token_text."""
        transliterations = {}
        unicodes = {}
        used_signs = set()

        for face in token_text.keys():
            lines = token_text[face]
            face_key = self.FACE_REMAPPING.get(face, face)

            # List of columns, each column is a list of lines
            face_transliterations: list[list[str]] = []
            face_unicodes: list[list[str]] = []

            current_column = {"transliteration": [], "unicode": []}

            for line in lines:
                if line == "<COL>":
                    if len(current_column["transliteration"]) > 0:
                        face_transliterations.append(current_column["transliteration"])
                    if len(current_column["unicode"]) > 0:
                        face_unicodes.append(current_column["unicode"])
                    current_column = {"transliteration": [], "unicode": []}
                    continue

                if type(line) == str:
                    continue

                used_signs.update(line.get("sign", ["<B>"]))

                current_column["transliteration"].append(line.get("raw", "<B>"))
                current_column["unicode"].append(" ".join(line.get("sign", ["<B>"])))

            if len(current_column["transliteration"]) > 0:
                face_transliterations.append(current_column["transliteration"])
            if len(current_column["unicode"]) > 0:
                face_unicodes.append(current_column["unicode"])

            if len(face_transliterations) == 1:
                # No need for column markers as there is only one column
                transliterations[face_key] = "\n".join(face_transliterations[0])
            else:
                transliterations[face_key] = "\n".join(
                    [
                        f"@column {i+1}\n" + "\n".join(column)
                        for i, column in enumerate(face_transliterations)
                    ]
                )

            if len(face_unicodes) == 1:
                # No need for column markers as there is only one column
                unicodes[face_key] = "\n".join(face_unicodes[0])
            else:
                unicodes[face_key] = "\n".join(
                    [
                        f"@column {i+1}\n" + "\n".join(column)
                        for i, column in enumerate(face_unicodes)
                    ]
                )

        return transliterations, unicodes, used_signs
