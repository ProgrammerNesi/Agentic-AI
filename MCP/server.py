from mcp.server.fastmcp import FastMCP

mcp = FastMCP("portfolio_server")

MY_INFO = {
    "name": "Nasir Yousuf",
    "college": "Jamia Millia Islamia",
    "branch": "B.Tech CSE (Data Science)",
    "cgpa": 8.95,
    "skills": [
        "Python",
        "C++",
        "FastAPI",
        "React",
        "Next.js",
        "LangChain",
        "LangGraph",
        "Google ADK",
        "RAG"
    ]
}


@mcp.tool()
async def get_profile() -> dict:
    """
    Return Nasir's profile information.
    """
    return MY_INFO


@mcp.tool()
async def get_name() -> str:
    """
    Return only the user's name.
    """
    return MY_INFO["name"]


@mcp.resource("profile://me")
async def profile_resource() -> str:
    return f"""
Name: {MY_INFO['name']}
College: {MY_INFO['college']}
Branch: {MY_INFO['branch']}
CGPA: {MY_INFO['cgpa']}
Skills: {", ".join(MY_INFO['skills'])}
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")