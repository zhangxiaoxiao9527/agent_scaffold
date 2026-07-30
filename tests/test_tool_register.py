import unittest

from agent_scaffold.tool_register import ToolRegistry


def add(a, b):
    return a + b


class ToolRegisterTests(unittest.TestCase):
    def test_register_and_get_tool(self):
        registry = ToolRegistry()

        spec = registry.register(
            "add",
            add,
            description="Add two numbers.",
            tags={"math"},
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )

        self.assertTrue(registry.has("add"))
        self.assertIs(registry.get("add"), spec)
        self.assertEqual(spec.to_dict()["tags"], ["math"])

    def test_search_by_query_and_tags(self):
        registry = ToolRegistry()
        registry.register("add", add, description="Add numbers.", tags={"math"})
        registry.register("say_hello", lambda name: f"hello {name}", tags={"text"})

        results = registry.search("add", tags={"math"})

        self.assertEqual([tool.name for tool in results], ["add"])

    def test_export_as_llm_tools(self):
        registry = ToolRegistry()
        registry.register("add", add, description="Add two numbers.")

        tools = registry.as_llm_tools()

        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "add")


if __name__ == "__main__":
    unittest.main()
