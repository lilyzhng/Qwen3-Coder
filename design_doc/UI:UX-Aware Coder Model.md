# The Core Problem
Today's coding agents generate generic, "developer-looking" UIs (random blues, purples, default Bootstrap/Tailwind palettes) because they were trained on code correctness, not design quality. To fix this, you need a 3-layer dataset strategy

# Datasets for Post-Training a UI/UX-Aware Coder Model

To train a model that generates production-ready, design-consistent UI code and avoids generic aesthetics, you need to move beyond standard code datasets. The key is to use a combination of datasets that teach the model not just *how* to code UIs, but *what* makes a UI good. Here’s a breakdown of the dataset types you should use and how to leverage them.


## Recommended Dataset Strategy: A 3-Layer Approach

No single dataset will solve this problem. The most effective strategy is to combine three types of data to create a comprehensive training set:

1.  **Component & Design System Datasets**: To learn consistency and structure.
2.  **Screenshot-to-Code Datasets**: To learn the visual representation of code.
3.  **Preference & Quality Datasets**: To learn what makes a design *good*.

### 1. Component & Design System Datasets

**Goal**: Teach the model to use consistent, reusable components and understand design tokens (colors, spacing, typography).

| Dataset/Source | Type | How to Use | Why It's Useful |
| :--- | :--- | :--- | :--- |
| **shadcn/ui** | Component Library | Scrape the component code and documentation. Create instruction-code pairs for each component and its variations. | Teaches the model to generate code that uses a well-structured, popular, and production-ready component library. The code is clean and follows best practices. |
| **Tailwind CSS Docs** | Documentation | Parse the official documentation to extract code examples for every utility class. | Provides a foundational understanding of the most popular CSS framework for modern web development. |
| **UIGEN-T1.1-TAILWIND** | Instruction-Code | A small but high-quality dataset of 805 instruction-code pairs using Tailwind CSS. Good starting point. | Directly teaches the model to follow natural language instructions to generate Tailwind CSS code. |
| **react-code-instructions** | Instruction-Code | 74.4K instruction-code pairs for React + Tailwind CSS. | A larger dataset for learning the relationship between instructions and React/Tailwind code. |
| **Public Design Systems** | Code & Docs | Scrape the code and documentation from open-source design systems like Material Design, Carbon Design System, or Fluent UI. | Teaches the model about design tokens, theming, and how to apply a consistent visual style across an entire application. |

### 2. Screenshot-to-Code Datasets

**Goal**: Connect the visual appearance of a UI to the underlying code. This is crucial for moving beyond generic designs.

| Dataset | Type | Size | How to Use | Why It's Useful |
| :--- | :--- | :--- | :--- | :--- |
| **WebCode2M** | Image-Code | 2.56M | Use the image-code pairs to train a multimodal model to generate code from a visual prompt. | Massive, real-world dataset that provides a strong foundation for understanding how HTML/CSS renders visually. |
| **WebSight** | Image-Code | 2M | Similar to WebCode2M, but synthetically generated. Can be used to augment training data. | Large-scale synthetic data can help improve model robustness. |
| **Design2Code** | Image-Code | 484 | A high-quality benchmark dataset. Use it for evaluation and for fine-tuning on a small set of very high-quality examples. | Real-world, manually curated examples of excellent web design. |
| **VISION2UI** | Image-Code | Not specified | Contains layout information, which is a critical component of good design. | Teaches the model to understand and replicate layout structures. |

### 3. Preference & Quality Datasets

**Goal**: This is the most important layer. It teaches the model to differentiate between good and bad design, and to align with user preferences. This is where you solve the "random blue and purple website" problem.

| Dataset | Type | Size | How to Use | Why It's Useful |
| :--- | :--- | :--- | :--- | :--- |
| **UICrit** | Quality Ratings & Critiques | 1K UIs, 11K critiques | **This is your key dataset.** Convert the ratings and critiques into preference pairs (chosen vs. rejected) for Direct Preference Optimization (DPO). For example, a UI with a high aesthetic score is a "chosen" example, and one with a low score is a "rejected" example. The natural language critiques can be used to generate detailed instructions for improvement. | Directly teaches the model what constitutes good and bad design, with both quantitative scores and qualitative feedback. |
| **AlignUI** | Preference Data | 720 preferences | The methodology is more important than the dataset itself. Replicate their crowdsourcing approach to build your own preference dataset focused on your specific design goals (e.g., brand consistency, accessibility). | Provides a framework for collecting nuanced user preferences (e.g., predictability, efficiency) that go beyond simple aesthetics. |
| **Awwwards/Dribbble** | Inspiration | N/A | Scrape award-winning designs from sites like Awwwards. While you won't have the code, you can use these images as positive examples in a multimodal preference model, or as visual prompts for a screenshot-to-code model to try and replicate. | Provides a source of constantly updated, high-quality, and creative web designs to keep your model's aesthetic sense current. |

## How to Modify and Combine These Datasets

1.  **Start with a Strong Base**: Pre-train or fine-tune your model on a large code dataset like **The Stack** or **StarCoder2** to give it a general understanding of code.

2.  **Instruction-Tune on UI Code**: Fine-tune the base model on a combination of **react-code-instructions**, **UIGEN-T1.1-TAILWIND**, and your own scraped data from **shadcn/ui** and other design systems. This will specialize the model for UI development.

3.  **Multimodal Fine-Tuning**: If you have the resources for a multimodal model, fine-tune it on **WebCode2M** and **WebSight**. This will connect the code generation to visual understanding.

4.  **Preference Tuning (DPO)**: This is the final and most critical step. Create a preference dataset from **UICrit** by pairing high-rated and low-rated designs. Augment this with your own preference data collected using the **AlignUI** methodology. Then, use DPO to fine-tune your model to generate designs that are not just syntactically correct, but also aesthetically pleasing and aligned with user preferences.

By following this layered approach, you can create a model that understands the nuances of good UI/UX design and generates code that is ready for production, not just a random assortment of colorful boxes.
