from lean_lsp_mcp.client_utils import startup_client
from mcp.server.fastmcp import Context

if __name__ == '__main__':
    ctx = Context()
    print(ctx)
    print(dir(ctx))
    startup_client(ctx)