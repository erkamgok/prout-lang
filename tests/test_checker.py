"""Validator: types, effects, preconditions, policy, and Money."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from support import ROOT, check, codes, snippet


def _money(expr: str, ret: str = "Money"):
    return check(
        f"""
        type Money
        flow f(a: Money, b: Money, i: Int, s: String) -> {ret}
          return {expr}
        """
    )


class TestTypes(unittest.TestCase):
    def test_unknown_type_field_variable_and_operation(self) -> None:
        diags = check(
            """
            type Box {
              id: Nope
            }
            flow f(box: Box) -> Int
              return box.missing
            """
        )
        self.assertIn("E-TYPE", codes(diags, "error"))
        self.assertIn("E-FIELD", codes(diags, "error"))

        diags = check(
            """
            flow f() -> Int
              return missing
            """
        )
        self.assertEqual(codes(diags, "error"), ["E-UNDEF"])

        diags = check(
            """
            flow f() -> Int
              return no.such()
            """
        )
        self.assertIn("E-UNDEF", codes(diags, "error"))

    def test_builtins_do_not_need_declarations(self) -> None:
        diags = check(
            """
            type Box {
              n: Int
              s: String
              b: Bool
            }
            flow f(box: Box) -> Bool
              return box.b
            """
        )
        self.assertEqual(diags, [])

    def test_argument_arity_and_return_mismatch(self) -> None:
        diags = check(
            """
            op id(x: Int, y: Int) -> Int
              impl native
            flow f(a: Money) -> String
              return id(a)
            type Money
            """
        )
        found = codes(diags, "error")
        self.assertIn("E-ARITY", found)
        self.assertIn("E-TYPE", found)

        diags = check(
            """
            op a() -> String
              impl native
            op b() -> Int
              impl a()
            """
        )
        self.assertIn("E-TYPE", codes(diags, "error"))

    def test_bool_operators_require_bool(self) -> None:
        self.assertEqual(
            check(
                """
                flow f(a: Bool, b: Bool) -> Bool
                  return a && b || a
                """
            ),
            [],
        )
        diags = check(
            """
            type Money
            flow f(a: Money, b: Money) -> Bool
              return a && b
            """
        )
        self.assertIn("E-TYPE", codes(diags, "error"))

    def test_pre_and_post_must_be_bool_and_posts_are_not_proved(self) -> None:
        diags = check(
            """
            op f(x: Int) -> Int
              pre x
              post 1
              impl native
            """
        )
        self.assertGreaterEqual(codes(diags, "error").count("E-TYPE"), 2)

        # A constant-false post is a claim about the impl, which is not executed.
        diags = check(
            """
            op f() -> Int
              post 1 > 2
              impl native
            flow g() -> Int
              return f()
            """
        )
        self.assertNotIn("E-PRE", codes(diags))
        self.assertEqual(diags, [])

    def test_post_may_mention_result(self) -> None:
        diags = check(
            """
            type Receipt {
              status: Status
            }
            type Status
            op f() -> Receipt
              post result.status == "refunded"
              impl native
            """
        )
        self.assertEqual(diags, [])

    def test_duplicates(self) -> None:
        diags = check(
            """
            type Money
            type Money
            op f() -> Int
              impl native
            op f() -> Int
              impl native
            flow g() -> Int
              let x = 1
              let x = 2
              return x
            flow g() -> Int
              return 1
            """
        )
        self.assertGreaterEqual(codes(diags, "error").count("E-DUP"), 4)

    def test_several_errors_are_reported_together(self) -> None:
        diags = check(
            """
            type Money
            type Order {
              total: Money
            }
            op inner() -> Int
              eff write(payments)
              impl native
            op pay(order: Order) -> Int
              eff net(paypal)
              impl inner()
            flow f(order: Order) -> String
              return order.total
            """
        )
        found = set(codes(diags, "error"))
        self.assertTrue({"E-EFF", "E-POLICY", "E-TYPE"} <= found)


class TestEffects(unittest.TestCase):
    def test_flow_effects_must_equal_the_union_of_calls(self) -> None:
        body = """
            op read_order() -> Int
              eff read(orders)
              impl native
            op charge() -> Int
              eff write(payments), net(stripe)
              impl native
            flow handle() -> Int
              eff {effects}
              let x = read_order()
              return charge()
        """
        self.assertEqual(check(body.format(effects="read(orders), write(payments), net(stripe)")), [])
        # Declaration order is not significant.
        self.assertEqual(check(body.format(effects="net(stripe), read(orders), write(payments)")), [])

        missing = check(body.format(effects="read(orders)"))
        self.assertIn("E-EFF", codes(missing, "error"))
        eff = next(d for d in missing if d.code == "E-EFF")
        self.assertIsNotNone(eff.fix)
        assert eff.fix is not None
        self.assertIn("read(orders)", eff.fix)
        self.assertIn("net(stripe)", eff.fix)
        self.assertIn("write(payments)", eff.fix)

        extra = check(body.format(effects="read(orders), write(payments), net(stripe), emit(audit)"))
        self.assertIn("E-EFF", codes(extra, "error"))

    def test_flow_with_no_effects_matches_an_empty_union(self) -> None:
        diags = check(
            """
            op pure() -> Int
              impl native
            flow f() -> Int
              let x = pure()
              return x
            """
        )
        self.assertEqual(diags, [])

    def test_unused_flow_effect_is_an_error_even_when_nothing_is_called(self) -> None:
        diags = check(
            """
            flow f() -> Int
              eff read(orders)
              return 1
            """
        )
        self.assertEqual(codes(diags, "error"), ["E-EFF"])

    def test_op_effects_must_match_a_called_impl(self) -> None:
        matched = check(
            """
            op inner() -> Int
              eff read(orders), net(stripe)
              impl native
            op outer() -> Int
              eff net(stripe), read(orders)
              impl inner()
            """
        )
        self.assertEqual(matched, [])

        missing = check(
            """
            op inner() -> Int
              eff read(orders), net(stripe)
              impl native
            op outer() -> Int
              eff read(orders)
              impl inner()
            """
        )
        self.assertIn("E-EFF", codes(missing, "error"))

        unused = check(
            """
            op inner() -> Int
              eff read(orders)
              impl native
            op outer() -> Int
              eff read(orders), write(payments)
              impl inner()
            """
        )
        self.assertIn("E-EFF", codes(unused, "error"))

    def test_native_effects_are_trusted(self) -> None:
        # `impl native` is a declaration. SPEC does not execute it.
        diags = check(
            """
            op pay() -> Int
              eff read(orders), write(payments), net(stripe), spend(stripe), emit(audit)
              impl native
            """
        )
        self.assertEqual(diags, [])

    def test_unknown_effect_kind_and_empty_target(self) -> None:
        diags = check(
            """
            op f() -> Int
              eff notify(slack), read()
              impl native
            """
        )
        self.assertGreaterEqual(codes(diags, "error").count("E-EFF"), 2)


class TestPreconditions(unittest.TestCase):
    def _refund(self, argument: str, extra_pre: str = "") -> list:
        # Built at column 0 so an extra `pre` line cannot disturb dedent.
        lines = [
            "type Money",
            "type Order {",
            "  total: Money",
            "  status: Status",
            "}",
            "type Status",
            "op refund(order: Order, amount: Money) -> Int",
            "  pre amount <= order.total",
        ]
        if extra_pre:
            lines.append(f"  {extra_pre}")
        lines += [
            "  impl native",
            "flow handle(order: Order) -> Int",
            f"  return refund(order, {argument})",
        ]
        from etsah import Checker, Parser

        return Checker(Parser("\n".join(lines) + "\n").parse()).check()

    def test_proved_precondition_is_silent(self) -> None:
        self.assertEqual(self._refund("order.total"), [])
        self.assertEqual(self._refund("order.total - 1"), [])

    def test_false_precondition_is_an_error_at_the_call(self) -> None:
        diags = self._refund("order.total + 1")
        errors = [d for d in diags if d.severity == "error"]
        self.assertEqual([d.code for d in errors], ["E-PRE"])
        self.assertEqual(errors[0].line, 11)
        self.assertTrue(errors[0].fix)

    def test_unknown_precondition_is_a_runtime_guard(self) -> None:
        diags = self._refund("order.total", 'pre order.status == "paid"')
        self.assertEqual(codes(diags, "error"), [])
        warnings = [d for d in diags if d.severity == "warning"]
        self.assertEqual([d.code for d in warnings], ["W-PRE"])
        self.assertEqual(warnings[0].line, 12)

    def test_false_and_unknown_are_both_reported(self) -> None:
        diags = self._refund("order.total + 100", 'pre order.status == "paid"')
        self.assertIn("E-PRE", codes(diags, "error"))
        self.assertIn("W-PRE", codes(diags, "warning"))

    def test_constant_preconditions(self) -> None:
        false_pre = check(
            """
            op f() -> Int
              pre 1 > 2
              impl native
            flow g() -> Int
              return f()
            """
        )
        self.assertEqual(codes(false_pre, "error"), ["E-PRE"])

        true_pre = check(
            """
            op f() -> Int
              pre 1 <= 2
              impl native
            flow g() -> Int
              return f()
            """
        )
        self.assertEqual(true_pre, [])

        same = check(
            """
            op f(x: Int) -> Int
              pre x == x
              impl native
            flow g() -> Int
              return f(1)
            """
        )
        self.assertEqual(same, [])

    def test_three_valued_and_or(self) -> None:
        def call(pre: str):
            return check(
                f"""
                type Order {{
                  status: Status
                }}
                type Status
                op f(order: Order) -> Int
                  pre {pre}
                  impl native
                flow g(order: Order) -> Int
                  return f(order)
                """
            )

        # false && unknown is false, so this is an error rather than a guard.
        self.assertIn("E-PRE", codes(call('false && order.status == "paid"'), "error"))
        self.assertNotIn("W-PRE", codes(call('false && order.status == "paid"'), "warning"))
        # true && unknown stays unknown.
        self.assertEqual(codes(call('true && order.status == "paid"'), "error"), [])
        self.assertIn("W-PRE", codes(call('true && order.status == "paid"'), "warning"))
        # true || unknown is proved.
        self.assertEqual(call('true || order.status == "paid"'), [])

    def test_division_is_not_constant_folded(self) -> None:
        diags = check(
            """
            op f() -> Int
              pre 1 / 2 <= 1
              impl native
            flow g() -> Int
              return f()
            """
        )
        self.assertEqual(codes(diags, "warning"), ["W-PRE"])
        self.assertEqual(codes(diags, "error"), [])

    def test_line_parser_precedence_feeds_the_checker(self) -> None:
        proved = check(
            """
            op f() -> Int
              pre 1 + 2 * 3 <= 7
              impl native
            flow g() -> Int
              return f()
            """
        )
        self.assertEqual(proved, [])
        false_pre = check(
            """
            op f() -> Int
              pre 1 + 2 * 3 <= 6
              impl native
            flow g() -> Int
              return f()
            """
        )
        self.assertEqual(codes(false_pre, "error"), ["E-PRE"])


class TestPolicy(unittest.TestCase):
    def _net(self, target: str, **kwargs):
        return check(
            f"""
            op pay() -> Int
              eff net({target})
              impl native
            """,
            **kwargs,
        )

    def test_default_policy_allows_only_stripe(self) -> None:
        self.assertEqual(self._net("stripe"), [])
        diags = self._net("paypal")
        self.assertEqual(codes(diags, "error"), ["E-POLICY"])
        self.assertEqual(diags[0].line, 1)

    def test_allow_list_replaces_the_default(self) -> None:
        self.assertEqual(self._net("paypal", allow_net=["paypal"]), [])
        diags = self._net("stripe", allow_net=["paypal"])
        self.assertEqual(codes(diags, "error"), ["E-POLICY"])

    def test_deny_wins_over_allow(self) -> None:
        diags = self._net("stripe", allow_net=["stripe"], deny_net=["stripe"])
        self.assertEqual(codes(diags, "error"), ["E-POLICY"])


class TestMoney(unittest.TestCase):
    def test_money_combines_with_money_and_int(self) -> None:
        cases = {
            "a + b": "Money",
            "a + i": "Money",
            "i + a": "Money",
            "a - b": "Money",
            "a - i": "Money",
            "a * i": "Money",
            "i * a": "Money",
            "a / i": "Money",
            "i + i": "Int",
            "i - i": "Int",
            "i * i": "Int",
            "i / i": "Int",
        }
        for expr, ret in cases.items():
            with self.subTest(expr=expr, ret=ret):
                self.assertEqual(_money(expr, ret), [], expr)

    def test_rejected_money_combinations(self) -> None:
        for expr in ("i - a", "a * b", "a / b", "i / a", "a + s", "s + a"):
            with self.subTest(expr=expr):
                self.assertIn("E-TYPE", codes(_money(expr), "error"), expr)

    def test_comparisons_are_bool_even_across_money_and_int(self) -> None:
        self.assertEqual(_money("a <= i", "Bool"), [])
        self.assertEqual(_money("a > b", "Bool"), [])
        diags = _money("a <= i", "Money")
        self.assertIn("E-TYPE", codes(diags, "error"))

    def test_refund_amount_uses_money_arithmetic(self) -> None:
        # order.total + 100 is still Money, and the comparison is decidable.
        diags = check(
            """
            type Money
            type Order {
              total: Money
            }
            op refund(order: Order, amount: Money) -> Int
              pre amount <= order.total
              impl native
            flow handle(order: Order) -> Int
              return refund(order, order.total + 100)
            """
        )
        self.assertEqual(codes(diags, "error"), ["E-PRE"])


class TestCli(unittest.TestCase):
    def _run(self, source: str, *args: str, dedent: bool = True) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.etsah"
            path.write_text(snippet(source) if dedent else source, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(ROOT / "etsah.py"), *args, str(path)],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_unreadable_file_exits_2_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_utf8 = Path(tmp) / "bad.etsah"
            bad_utf8.write_bytes(b"\xff\xfe\x00not utf-8")
            targets = [Path(tmp) / "no-such-file.etsah", Path(tmp), bad_utf8]
            for cmd in ("check", "ast"):
                for target in targets:
                    with self.subTest(cmd=cmd, target=target.name):
                        result = subprocess.run(
                            [sys.executable, str(ROOT / "etsah.py"), cmd, str(target)],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                        self.assertNotIn("Traceback", result.stderr)
                        self.assertTrue(result.stderr.startswith("[ERROR] "), result.stderr)

    def test_exit_codes(self) -> None:
        ok = self._run(
            """
            op f() -> Int
              impl native
            flow g() -> Int
              return f()
            """,
            "check",
        )
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)

        warning = self._run(
            """
            type Status
            type Order {
              status: Status
            }
            op f(order: Order) -> Int
              pre order.status == "paid"
              impl native
            flow g(order: Order) -> Int
              return f(order)
            """,
            "check",
            "--json",
        )
        self.assertEqual(warning.returncode, 0, warning.stdout + warning.stderr)
        payload = json.loads(warning.stdout)
        self.assertEqual(payload["diagnostics"][0]["severity"], "warning")
        self.assertEqual(payload["diagnostics"][0]["code"], "W-PRE")
        self.assertEqual(len(payload["hash"]), 12)

        failed = self._run(
            """
            flow g() -> Int
              return "no"
            """,
            "check",
        )
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)

        broken = self._run("  type Broken\n", "check", dedent=False)
        self.assertEqual(broken.returncode, 2, broken.stderr)
        self.assertIn("line", broken.stderr)

    def test_ai_task_examples_pin_check_json_keys(self) -> None:
        passing = subprocess.run(
            [sys.executable, str(ROOT / "etsah.py"), "check", "--json", str(ROOT / "examples" / "ai_tasks.etsah")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(passing.returncode, 0, passing.stdout + passing.stderr)
        ok = json.loads(passing.stdout)
        self.assertEqual(set(ok), {"hash", "diagnostics", "audit_id", "plan", "manifest"})
        self.assertEqual(len(ok["hash"]), 12)
        self.assertEqual(len(ok["audit_id"]), 64)
        self.assertIsInstance(ok["plan"], list)
        self.assertTrue(ok["plan"])
        self.assertEqual(ok["diagnostics"], [])

        failing = subprocess.run(
            [sys.executable, str(ROOT / "etsah.py"), "check", "--json", str(ROOT / "examples" / "ai_tasks_broken.etsah")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(failing.returncode, 1, failing.stdout + failing.stderr)
        bad = json.loads(failing.stdout)
        self.assertEqual(set(bad), {"hash", "diagnostics", "audit_id", "plan", "manifest"})
        errors = [item for item in bad["diagnostics"] if item["severity"] == "error"]
        self.assertTrue(errors)
        self.assertTrue(all(item["code"] == "E-POLICY" for item in errors))
        allowed = {"code", "severity", "line", "message", "fix", "detail", "context"}
        for item in bad["diagnostics"]:
            self.assertTrue({"code", "severity", "line", "message"} <= set(item))
            self.assertTrue(set(item) <= allowed)

    def test_ast_command_prints_canonical_json(self) -> None:
        proc = self._run(
            """
            type Money
            type Order {
              total: Money
            }
            """,
            "ast",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual([t["name"] for t in data["types"]], ["Money", "Order"])

    def test_allow_net_flag(self) -> None:
        source = """
            op pay() -> Int
              eff net(paypal)
              impl native
        """
        denied = self._run(source, "check")
        self.assertEqual(denied.returncode, 1)
        allowed = self._run(source, "check", "--allow-net", "paypal")
        self.assertEqual(allowed.returncode, 0, allowed.stdout + allowed.stderr)


if __name__ == "__main__":
    unittest.main()
