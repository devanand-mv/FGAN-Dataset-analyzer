import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from data_processor import TabularDataProcessor
from gan_model import TabularGAN
from evaluator import ModelEvaluator

# Set page config
st.set_page_config(
    page_title="Tabular GAN Data Balancer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .team-badge {
        background: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1E293B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .balanced-box {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 16px;
        color: #065F46;
        margin-bottom: 1rem;
    }
    .imbalanced-box {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 16px;
        color: #92400E;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title & Proposal Header
st.markdown('<div class="main-title">🧬 GAN-Based Synthetic Data Generation for Imbalanced Datasets</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A web-based tool for class distribution analysis and GAN-assisted tabular data balancing</div>', unsafe_allow_html=True)
st.markdown("""
<div class="team-badge">
    <strong>Project Team:</strong> DJOHAN CHERUSSERIL (URK24CS7147) &nbsp;|&nbsp; DEVANAND M V (URK24CS7111) &nbsp;|&nbsp; LAYO BAIJU (URK24CS7128)
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: DATASET INPUT ---
st.sidebar.header("📁 Step 1: Provide Dataset")

sample_datasets = {
    "Upload Your Own CSV": None,
    "Sample 1: Credit Default (Imbalanced ~12% Minority)": "sample_data/credit_default_imbalanced.csv",
    "Sample 2: Medical Diagnosis (Imbalanced ~18% Minority)": "sample_data/medical_diagnosis_imbalanced.csv",
    "Sample 3: Balanced Dataset (50:50 Parity)": "sample_data/balanced_dataset_sample.csv"
}

data_choice = st.sidebar.selectbox("Choose Data Source", list(sample_datasets.keys()))

df_raw = None
if data_choice == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV tabular dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
else:
    file_path = sample_datasets[data_choice]
    if os.path.exists(file_path):
        df_raw = pd.read_csv(file_path)
        st.sidebar.info(f"Loaded built-in: {os.path.basename(file_path)}")

if df_raw is None:
    st.info("👈 Please upload a CSV file or select one of the built-in benchmark datasets from the sidebar to begin.")
    st.stop()

# Target Column Selection
st.sidebar.markdown("---")
st.sidebar.header("🎯 Target / Class Column")

# Auto-detect target column if common name matches
possible_targets = [c for c in df_raw.columns if any(k in c.lower() for k in ['target', 'class', 'default', 'outcome', 'label', 'status'])]
default_idx = df_raw.columns.get_loc(possible_targets[0]) if possible_targets else len(df_raw.columns) - 1

target_col = st.sidebar.selectbox(
    "Select the classification label column:",
    options=df_raw.columns,
    index=int(default_idx)
)

# Initialize Processor & Analyze Balance
processor = TabularDataProcessor(target_col=target_col)
imbalance_info = processor.analyze_distribution(df_raw)

# Dataset overview in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset Rows:** `{df_raw.shape[0]}`")
st.sidebar.markdown(f"**Dataset Columns:** `{df_raw.shape[1]}`")
st.sidebar.markdown(f"**Distinct Classes:** `{len(imbalance_info['classes'])}`")
st.sidebar.markdown(f"**Imbalance Ratio:** `{imbalance_info['imbalance_ratio']} : 1`")

# --- MAIN CONTENT: DATASET PREVIEW ---
with st.expander("👀 View Raw Dataset Preview & Schema", expanded=False):
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.dataframe(df_raw.head(8), use_container_width=True)
    with col_p2:
        st.markdown("**Data Types:**")
        st.dataframe(pd.DataFrame(df_raw.dtypes, columns=["Dtype"]).astype(str), use_container_width=True)

st.markdown("---")

# --- USER DECISION SECTION: THREE CORE OPTIONS ---
st.subheader("🛠️ What would you like to do with this dataset?")
st.markdown("Select an action below to inspect balance status, balance using PyTorch GAN, or keep the original dataset as-is:")

action_mode = st.radio(
    label="Choose Action:",
    options=[
        "Option 1: Check if Dataset is Balanced (Analysis & Diagnostic)",
        "Option 2: Balance Dataset using Generative Adversarial Network (GAN)",
        "Option 3: Leave Dataset As-Is (Proceed without Balancing)"
    ],
    index=0
)

st.markdown("---")

# ==============================================================================
# OPTION 1: CHECK BALANCE
# ==============================================================================
if "Option 1" in action_mode:
    st.header("📊 Class Distribution & Imbalance Check")

    # Display clear banner according to balance state
    if imbalance_info["status"] == "BALANCED":
        st.markdown(f"""
        <div class="balanced-box">
            <h3 style="margin-top:0;">✅ Dataset is BALANCED!</h3>
            <p style="font-size: 1.05rem; margin-bottom: 0;">
                {imbalance_info['status_desc']}<br>
                Both classes are adequately represented. Standard machine learning classifiers can be trained directly without oversampling or synthetic augmentation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="imbalanced-box">
            <h3 style="margin-top:0;">⚠️ Dataset is NOT BALANCED!</h3>
            <p style="font-size: 1.05rem; margin-bottom: 0;">
                {imbalance_info['status_desc']}<br>
                <strong>Majority Class ({imbalance_info['maj_class']}):</strong> {imbalance_info['maj_count']} samples ({imbalance_info['percentages'][0]}%)<br>
                <strong>Minority Class ({imbalance_info['min_class']}):</strong> {imbalance_info['min_count']} samples ({imbalance_info['percentages'][-1]}%)<br>
                <strong>Deficit to achieve 50:50 parity:</strong> Need <strong>{imbalance_info['num_needed_for_50_50']}</strong> additional minority samples.<br>
                👉 <em>You can switch to <strong>Option 2</strong> to synthesize realistic minority records with a PyTorch GAN!</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Metrics Summary Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Records", f"{imbalance_info['total_samples']:,}")
    with col_m2:
        st.metric("Imbalance Ratio", f"{imbalance_info['imbalance_ratio']} : 1")
    with col_m3:
        st.metric(f"Majority Class '{imbalance_info['maj_class']}'", f"{imbalance_info['maj_count']} ({imbalance_info['percentages'][0]}%)")
    with col_m4:
        st.metric(f"Minority Class '{imbalance_info['min_class']}'", f"{imbalance_info['min_count']} ({imbalance_info['percentages'][-1]}%)")

    # Interactive Plots
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_bar = px.bar(
            x=[str(c) for c in imbalance_info["classes"]],
            y=imbalance_info["counts"],
            color=[str(c) for c in imbalance_info["classes"]],
            labels={"x": f"Target: {target_col}", "y": "Record Count"},
            title="Class Frequency Distribution",
            text=imbalance_info["counts"],
            color_discrete_sequence=["#3B82F6", "#EF4444", "#10B981"]
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_c2:
        fig_pie = px.pie(
            names=[f"Class {c}" for c in imbalance_info["classes"]],
            values=imbalance_info["counts"],
            title="Class Proportion Share (%)",
            color_discrete_sequence=["#3B82F6", "#EF4444", "#10B981"],
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ==============================================================================
# OPTION 2: BALANCE DATASET WITH GAN
# ==============================================================================
elif "Option 2" in action_mode:
    st.header("🧬 GAN-Based Synthetic Minority Generation")

    if imbalance_info["status"] == "BALANCED":
        st.warning("ℹ️ Notice: This dataset is already balanced! You can still run the GAN synthesizer to test generative capabilities, but balancing is not required for learning.")

    st.markdown("""
    This module trains a **PyTorch Tabular Generative Adversarial Network (GAN)** on real minority-class instances:
    - **Generator Network:** Translates Gaussian latent noise into realistic continuous & encoded tabular attributes.
    - **Discriminator Network:** Adversarially learns to discern real minority profiles from synthetic fakes.
    - **Post-Synthesis Reconstruction:** Maps continuous GAN outputs back to original data types, valid domain bounds, and categories.
    """)

    # Configuration Parameters
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.subheader("1. Balancing Goal")
        balance_mode = st.radio(
            "Select Target Strategy:",
            ["Full Balance (50:50 Parity)", "Custom Synthetic Sample Count"]
        )
        if balance_mode == "Full Balance (50:50 Parity)":
            samples_to_generate = max(50, imbalance_info["num_needed_for_50_50"])
            st.info(f"Will synthesize **{samples_to_generate}** records of minority class **'{imbalance_info['min_class']}'**.")
        else:
            samples_to_generate = st.slider(
                "Number of synthetic records to generate:",
                min_value=20,
                max_value=max(500, imbalance_info["total_samples"]),
                value=max(50, imbalance_info["num_needed_for_50_50"]),
                step=10
            )

    with col_cfg2:
        st.subheader("2. GAN Training Settings")
        with st.expander("⚙️ Hyperparameters (PyTorch)", expanded=True):
            epochs = st.slider("Training Epochs", min_value=50, max_value=300, value=120, step=10)
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
            latent_dim = st.select_slider("Latent Noise Dimension ($z$)", options=[16, 32, 64], value=32)
            lr = st.select_slider("Learning Rate", options=[0.0001, 0.0002, 0.0005, 0.001], value=0.0002)

    # Train / Generate Action Button
    st.markdown("---")
    train_btn = st.button("🚀 Train GAN & Synthesize Minority Data", type="primary", use_container_width=True)

    if train_btn:
        # Separate Train and Untouched Test Set to prevent data leakage (Project Proposal Section 10)
        train_df, test_df = train_test_split(df_raw, test_size=0.25, random_state=42, stratify=df_raw[target_col])
        
        # Fit processor on training features
        min_class = imbalance_info["min_class"]
        train_minority = train_df[train_df[target_col] == min_class]
        
        processor_gan = TabularDataProcessor(target_col=target_col)
        # Process full train set to learn scalers & bounds
        train_features_scaled = processor_gan.fit_transform(train_df, is_training=True)
        
        # Extract only minority feature representations for GAN training
        minority_idx = (train_df[target_col] == min_class).values
        minority_data_scaled = train_features_scaled[minority_idx]
        
        st.write(f"🔬 Training on **{len(minority_data_scaled)}** real training records of class `{min_class}` across **{minority_data_scaled.shape[1]}** features.")

        # Progress elements
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Live loss plot container
        plot_spot = st.empty()

        # Initialize PyTorch GAN
        gan = TabularGAN(
            input_dim=minority_data_scaled.shape[1],
            latent_dim=latent_dim,
            hidden_dim=128,
            lr=lr
        )

        def on_epoch_update(epoch, total_epochs, d_loss, g_loss):
            pct = epoch / total_epochs
            progress_bar.progress(pct)
            status_text.markdown(f"**Epoch {epoch}/{total_epochs}** &nbsp;|&nbsp; 📉 Discriminator Loss: `{d_loss:.4f}` &nbsp;|&nbsp; 📈 Generator Loss: `{g_loss:.4f}`")

        # Train GAN
        with st.spinner("Training GAN adversarially..."):
            history = gan.train(
                minority_data_scaled,
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=on_epoch_update
            )
            
        progress_bar.progress(1.0)
        status_text.success("🎉 GAN training successfully completed!")

        # Display Final Loss Curve
        loss_df = pd.DataFrame(history)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=loss_df["epoch"], y=loss_df["d_loss"], mode='lines', name='Discriminator Loss', line=dict(color='#EF4444', width=2)))
        fig_loss.add_trace(go.Scatter(x=loss_df["epoch"], y=loss_df["g_loss"], mode='lines', name='Generator Loss', line=dict(color='#3B82F6', width=2)))
        fig_loss.update_layout(title="Adversarial Loss Convergence (PyTorch)", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_white")
        st.plotly_chart(fig_loss, use_container_width=True)

        # Generate Synthetic Samples
        with st.spinner(f"Sampling {samples_to_generate} synthetic instances from latent space..."):
            synthetic_matrix = gan.generate(samples_to_generate)
            synthetic_df = processor_gan.inverse_transform(synthetic_matrix)
            synthetic_df[target_col] = min_class

        # Assemble Balanced Dataset (Original Train + Synthetic Minority)
        balanced_train_df = pd.concat([train_df, synthetic_df], ignore_index=True)

        st.session_state["balanced_df"] = balanced_train_df
        st.session_state["synthetic_df"] = synthetic_df
        st.session_state["train_df"] = train_df
        st.session_state["test_df"] = test_df
        st.session_state["processor_gan"] = processor_gan

    # If results are generated in session state, show evaluation and download
    if "synthetic_df" in st.session_state:
        synthetic_df = st.session_state["synthetic_df"]
        balanced_train_df = st.session_state["balanced_df"]
        train_df = st.session_state["train_df"]
        test_df = st.session_state["test_df"]
        processor_gan = st.session_state["processor_gan"]

        st.markdown("---")
        st.header("📊 Results: Before vs. After GAN Balancing")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            orig_counts = train_df[target_col].value_counts()
            fig_before = px.bar(
                x=[str(c) for c in orig_counts.index],
                y=orig_counts.values,
                title="Training Set: BEFORE GAN Balancing",
                labels={"x": target_col, "y": "Record Count"},
                text=orig_counts.values,
                color_discrete_sequence=["#EF4444"]
            )
            fig_before.update_traces(textposition='outside')
            st.plotly_chart(fig_before, use_container_width=True)

        with col_res2:
            bal_counts = balanced_train_df[target_col].value_counts()
            fig_after = px.bar(
                x=[str(c) for c in bal_counts.index],
                y=bal_counts.values,
                title="Training Set: AFTER GAN Balancing",
                labels={"x": target_col, "y": "Record Count"},
                text=bal_counts.values,
                color_discrete_sequence=["#10B981"]
            )
            fig_after.update_traces(textposition='outside')
            st.plotly_chart(fig_after, use_container_width=True)

        # Preview Synthetic Samples
        st.subheader("🔍 Synthetic Records Generated by GAN")
        st.dataframe(synthetic_df.head(10), use_container_width=True)

        # Synthetic Quality & Diversity Checks (Section 10 of proposal)
        st.subheader("🛡️ Synthetic Quality & Diversity Validation")
        real_min_df = train_df[train_df[target_col] == imbalance_info["min_class"]].drop(columns=[target_col])
        syn_min_df = synthetic_df.drop(columns=[target_col])
        quality = ModelEvaluator.evaluate_quality(real_min_df, syn_min_df, processor_gan.numerical_cols)

        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            st.metric("Generated Samples", quality["total_synthetic"])
        with col_q2:
            st.metric("Exact Overlap with Real", f"{quality['exact_overlap_with_real']} ({quality['overlap_percentage']}%)")
        with col_q3:
            st.metric("Synthetic Diversity", f"{quality['synthetic_diversity_unique_pct']}% unique")
        with col_q4:
            st.metric("Missing / Null Values", quality["null_count"])

        st.caption("Lower overlap with real data confirms the GAN is generalizing rather than memorizing training instances.")

        # Statistical Moments Comparison Table
        with st.expander("📈 Real vs. Synthetic Feature Statistics (Mean & Std Deviation)", expanded=False):
            st.dataframe(quality["feature_stats"], use_container_width=True)

        # Downstream Classifier Comparison (Proposal Section 10)
        st.markdown("---")
        st.header("🤖 Downstream Machine Learning Utility Assessment")
        st.markdown("""
        To verify real-world utility without test contamination, we train a classifier on:
        1. **Original Imbalanced Data**
        2. **GAN-Balanced Data**
        Both models are evaluated strictly on the **same untouched real test set**.
        """)

        clf_type = st.selectbox("Choose Evaluation Classifier:", ["RandomForest", "LogisticRegression"])

        # Prepare X and y
        feature_cols = processor_gan.feature_cols
        X_train_orig = train_df[feature_cols]
        y_train_orig = train_df[target_col]

        X_train_bal = balanced_train_df[feature_cols]
        y_train_bal = balanced_train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # One-hot encode any remaining categoricals for sklearn
        X_train_orig_enc = pd.get_dummies(X_train_orig)
        X_train_bal_enc = pd.get_dummies(X_train_bal)
        X_test_enc = pd.get_dummies(X_test)

        # Align columns
        all_cols = list(set(X_train_orig_enc.columns).union(X_train_bal_enc.columns).union(X_test_enc.columns))
        X_train_orig_enc = X_train_orig_enc.reindex(columns=all_cols, fill_value=0)
        X_train_bal_enc = X_train_bal_enc.reindex(columns=all_cols, fill_value=0)
        X_test_enc = X_test_enc.reindex(columns=all_cols, fill_value=0)

        eval_results = ModelEvaluator.train_and_compare_classifiers(
            X_train_orig_enc, y_train_orig,
            X_train_bal_enc, y_train_bal,
            X_test_enc, y_test,
            model_type=clf_type,
            minority_class=imbalance_info["min_class"]
        )

        st.dataframe(eval_results["comparison_df"], use_container_width=True)

        # Confusion Matrices Display
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown("**Original Imbalanced Model Confusion Matrix:**")
            cm_orig = np.array(eval_results["original"]["Confusion Matrix"])
            fig_cm_orig = px.imshow(cm_orig, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm_orig, use_container_width=True)

        with cm_col2:
            st.markdown("**GAN-Balanced Model Confusion Matrix:**")
            cm_bal = np.array(eval_results["balanced"]["Confusion Matrix"])
            fig_cm_bal = px.imshow(cm_bal, text_auto=True, color_continuous_scale="Greens", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm_bal, use_container_width=True)

        # Export Controls
        st.markdown("---")
        st.subheader("📥 Export & Download Results")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_balanced = balanced_train_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Full Balanced Dataset (CSV)",
                data=csv_balanced,
                file_name="balanced_dataset_gan.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            csv_syn_only = synthetic_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Synthetic Samples Only (CSV)",
                data=csv_syn_only,
                file_name="synthetic_minority_samples.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==============================================================================
# OPTION 3: LEAVE AS-IS
# ==============================================================================
elif "Option 3" in action_mode:
    st.header("📁 Leave Dataset As-Is (No Modification)")
    st.info("You selected to retain the original dataset distribution without applying GAN synthetic oversampling.")

    st.markdown("""
    ### Why keep the dataset as-is?
    - **Natural Class Frequency:** In some domains (e.g. disease diagnosis, rare event modeling), the natural rarity of the minority class is an authentic characteristic of the phenomenon.
    - **Algorithmic Alternative:** You can apply cost-sensitive learning or focal loss instead of resampling.
    - **Baseline Establishment:** Evaluates the baseline performance before attempting any generative augmentation.
    """)

    # Baseline Evaluation on Unmodified Data
    st.subheader("🤖 Evaluate Baseline Model on Original Imbalanced Data")
    if st.button("Run Baseline Classifier on Original Data"):
        train_df, test_df = train_test_split(df_raw, test_size=0.25, random_state=42, stratify=df_raw[target_col])
        X_train = pd.get_dummies(train_df.drop(columns=[target_col]))
        y_train = train_df[target_col]
        X_test = pd.get_dummies(test_df.drop(columns=[target_col]))
        y_test = test_df[target_col]

        all_cols = list(set(X_train.columns).union(X_test.columns))
        X_train = X_train.reindex(columns=all_cols, fill_value=0)
        X_test = X_test.reindex(columns=all_cols, fill_value=0)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        st.text("Classification Report (Original Untouched Data):")
        st.code(classification_report(y_test, preds))

    csv_original = df_raw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Unmodified Dataset (CSV)",
        data=csv_original,
        file_name="original_dataset_unmodified.csv",
        mime="text/csv"
    )
