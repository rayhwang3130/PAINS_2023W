def train(model, train_data, hyperparameters, **kwargs):

    model.fit(
        train_data=train_data,
        # hyperparameters = hyperparameters,
        **kwargs
    )

    return model
