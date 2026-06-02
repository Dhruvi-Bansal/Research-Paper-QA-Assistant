import os


def save_uploaded_files(uploaded_files):

    os.makedirs(
        "data/papers",
        exist_ok=True
    )

    file_paths = []

    for file in uploaded_files:

        path = os.path.join(
            "data/papers",
            file.name
        )

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        file_paths.append(path)

    return file_paths