import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xllmx",
    version="0.0.1",
    author="Alpha-VLLM",
    description="An Open-source Toolkit for LLM-centered Any2Any Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alpha-VLLM/Lumina-mGPT",
    packages=["xllmx"],
    include_package_data=True,
)
