import streamlit as st


def app():
    st.title("WikEd Error Corpus Preview")
    st.write(
        "[Paper](https://emjotde.github.io/publications/pdf/mjd.poltal2014.draft.pdf)"
    )
    st.write(
        """
        ノイズが多くてクリーニングちゃんとしないと使えなそう。

        ```
        wiked-v1.0.en.prepro
        ├── README.md
        ├── wiked.tok.cor
        └── wiked.tok.err
        ```

        データサイズ：28,588,505行, 3.5GB * 2

        wiked.tok.err
        ```
        Organic chemistry - Computer programming - Computer imaging - Cookbook
        Organic chemistry - Computer programming - Computer imaging - Cookbook - How to be Environmental
        The Wikimedia Free Textbook Project is a subsection of the Wikipedia set up on July 10 , 2003 for the cooperative development of open content textbooks of various fields and topics .
        Sister Project .
        Math and sciences : - Organic chemistry - Cell biology - Introductory physics - Electromagnetic Field Theory - Linear algebra - Environmental Awareness - Calculus
        Math and sciences : - Organic chemistry - Cell biology - Introductory physics - Physics - Linear algebra - Environmental Awareness - Calculus - Applied mathematics
        Math and sciences : - Organic chemistry - Cell biology - Introductory physics - Physics - Linear algebra - Environmental Awareness - Calculus - Applied mathematics - Optics
        Sciences : - Organic Chemistry - Physics - Cell Biology - Environmental Awareness
        Sciences : - Organic Chemistry - Physics - Cell Biology - Environmental Awareness - Horticulture
        Languages : - Spanish - Lojban
        ```

        wiked.tok.err
        ```
        Organic chemistry - Computer programming - Computer imaging
        Organic chemistry - Computer programming - Computer imaging - Cookbook - How to be &apos; Green&apos;
        The Wikimedia Free Textbook Project is a subsection of the Wikipedia set up for the cooperative development of open content textbooks of various fields and topics .
        Sister Wikis .
        Math and sciences : - Organic chemistry - Cell biology - Introductory physics - Electromagnetic Field Theory - Linear algebra - Environmental Awareness
        Math and sciences : - Organic chemistry - Cell biology - Introductory physics - Electromagnetic Field Theory - Linear algebra - Environmental Awareness - Calculus - Applied mathematics
        Math and sciences : - Organic chemistry - Cell biology - Introductory physics - Physics - Linear algebra - Environmental Awareness - Calculus - Applied mathematics
        Sciences : - Organic Chemistry - Physics - Optics - Cell Biology - Environmental Awareness
        Sciences : - Organic Chemistry - Physics - Cell Biology - Environmental Awareness
        Languages : - Spanish
        ```

        """
    )

