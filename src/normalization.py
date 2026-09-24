import re
import unicodedata


def normalize_text(text):
    """
    Basic Unicode-safe normalization for business names and addresses.
    """

    if text is None:
        return ""

    # Handle missing values
    if str(text).lower() == "nan":
        return ""

    text = str(text)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Standardize ampersand
    text = text.replace("&", " and ")

    # Keep Unicode letters/numbers and remove punctuation
    text = "".join(
    char if (
        char.isalnum()
        or char.isspace()
        or unicodedata.category(char).startswith("M")
    )
    else " "
    for char in text
)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(name):
    """
    Normalize a business name.
    """
    return normalize_text(name)


def normalize_address(address):
    """
    Normalize an address while preserving numbers.
    """
    return normalize_text(address)


def normalize_country(country):
    """
    Normalize country values.
    """
    return normalize_text(country)


def normalize_dataframe(df):
    """
    Add normalized versions of relevant columns
    while preserving the original columns.
    """

    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].apply(normalize_business_name)
    )

    df["business_address_normalized"] = (
        df["business_address"].apply(normalize_address)
    )

    df["country_normalized"] = (
        df["country"].apply(normalize_country)
    )

    return df


if __name__ == "__main__":

    examples = [
        "Orelee's Barbershop",
        "B+ Retail Inc",
        "ABC & Sons",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Pvt. EFS Print Ventures Ltd.",
        "797, Lake Town Block A, Kolkata, Howrah, West Bengal",
    ]

    for example in examples:
        print(f"{example} -> {normalize_text(example)}")