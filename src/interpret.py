
    # -------------------------
    # 9. Model Interpretion
    # -------------------------

    best_pipeline = search.best_estimator_

    model = best_pipeline.named_steps["model"]
    feature_names = (
        best_pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    # Impurity importance
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(
        "importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance_df)


    # Permutation importance
    perm_importance = permutation_importance(
        best_pipeline,
        X_train,
        y_train,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    permutation_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values(
        "importance_mean",
        ascending=False
    )

    print("\nPermutation Importance:")
    print(permutation_df)
    
    # -------------------------
    # SHAP interpretation
    # -------------------------

    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    # Transform training data
    X_train_transformed = preprocessor.transform(X_train)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrame
    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index
    )

    # Use a sample of 1,000 observations for SHAP
    X_shap = X_train_transformed.sample(
        n=1000,
        random_state=42
    )

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_shap)

    print("\nSHAP calculation completed.")

    print("SHAP values type:")
    print(type(shap_values))

    print("SHAP values shape:")
    print(np.array(shap_values).shape)

    # SHAP values for Bad class
    shap_values_bad = shap_values[:, :, 1]

    # Global SHAP summary plot
    shap.summary_plot(
        shap_values_bad,
        X_shap,
        show=True
    )