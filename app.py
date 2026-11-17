
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # 4. CNN Explainability - Permutation Feature Importance
                st.subheader("4. CNN Explainability - Permutation Feature Importance")

                importance_columns = ["Feature", "Mean F1 Score Drop"]
                if all(column in importance_df.columns for column in importance_columns):
                    top_features = importance_df.head(10).sort_values(
                        "Mean F1 Score Drop", ascending=True
                    )
                    fig, ax = plt.subplots(figsize=(9, 5))
                    x_errors = (
                        top_features["Std F1 Score Drop"]
                        if "Std F1 Score Drop" in top_features.columns
                        else None
                    )
                    ax.barh(
                        top_features["Feature"],
                        top_features["Mean F1 Score Drop"],
                        xerr=x_errors,
                        capsize=3
                    )
                    ax.axvline(0, linewidth=1)
                    ax.set_xlabel("Decrease in F1-score after shuffling")
                    ax.set_ylabel("Input Feature")
                    ax.set_title("CNN Permutation Feature Importance")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.dataframe(importance_df.head(10).round(4), use_container_width=True)
                else:
                    st.warning("The feature-importance data has an unexpected format.")

                # 5. CNN Model Loss
                st.subheader("5. CNN Model Loss")

                if "loss" in history_data:
                    epochs = range(1, len(history_data["loss"]) + 1)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(epochs, history_data["loss"], label="Training Loss")
                    if "val_loss" in history_data:
                        ax.plot(epochs, history_data["val_loss"], label="Validation Loss")
                    ax.set_title("CNN Training and Validation Loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Training loss history is not available.")

                # 6. CNN Model Accuracy
                st.subheader("6. CNN Model Accuracy")

                if "accuracy" in history_data:
                    epochs = range(1, len(history_data["accuracy"]) + 1)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(epochs, history_data["accuracy"], label="Training Accuracy")
                    if "val_accuracy" in history_data:
                        ax.plot(epochs, history_data["val_accuracy"], label="Validation Accuracy")
                    ax.set_title("CNN Training and Validation Accuracy")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Accuracy")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Training accuracy history is not available.")

        except Exception as error:

            st.error("Unable to load or display the model evaluation results.")
            st.code(str(error))


# =====================================================
# 14. ABOUT PAGE
# =====================================================

elif page == "About":

    st.header("About the Project")

    st.write(
        """
        The Credit Card Fraud Detection System is an
        Artificial Intelligence application developed
        using Deep Learning.

        It uses a Convolutional Neural Network (CNN)
        trained on credit card transaction data.

        The purpose of this system is to identify
        potentially fraudulent transactions and
        distinguish them from normal transactions.
        """
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - TensorFlow
        - Keras
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        """
    )

    st.warning(
        "This application is an educational project. "
        "Its predictions should not be used as the sole "
        "basis for real financial decisions."
    )


# =====================================================
# 15. FOOTER
0.25
