/**
 * Every exported item in `src` carries a doc comment.
 *
 * This package ships its source as well as its build (`package.json`
 * `files`), and a reader of either sees the comment, never only the
 * signature. Checked with the compiler's own parser, which already knows
 * which member is public and which overload signature is the one a caller
 * never sees, rather than guessing at either from a regular expression.
 */

import { readdirSync, readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

const SRC = new URL("../src/", import.meta.url);

interface Missing {
  readonly file: string;
  readonly line: number;
  readonly name: string;
}

/** Plain text out of a JSDoc comment value, or null when there is none. */
function textOf(comment: string | ts.NodeArray<ts.JSDocComment> | undefined): string | null {
  if (comment === undefined) return null;
  const text = typeof comment === "string" ? comment : ts.getTextOfJSDocComment(comment);
  return text !== undefined && text.trim() !== "" ? text.trim() : null;
}

/**
 * The text of a node's own doc comment, or null when it has none or it is
 * blank. A tag's own text, such as `@internal`'s, counts the same as the
 * main comment's: both are prose inside the one block comment a reader sees.
 */
function docOf(node: ts.Node): string | null {
  for (const entry of ts.getJSDocCommentsAndTags(node)) {
    const own = textOf((entry as { comment?: string | ts.NodeArray<ts.JSDocComment> }).comment);
    if (own !== null) return own;
    for (const tag of (entry as { tags?: readonly ts.JSDocTag[] }).tags ?? []) {
      const fromTag = textOf(tag.comment);
      if (fromTag !== null) return fromTag;
    }
  }
  return null;
}

function isExported(node: ts.Node): boolean {
  return (ts.getCombinedModifierFlags(node as unknown as ts.Declaration) & ts.ModifierFlags.Export) !== 0;
}

/** Private, protected, or `#` named: never part of the surface a caller sees. */
function isPrivateLike(member: ts.ClassElement): boolean {
  const flags = ts.getCombinedModifierFlags(member as unknown as ts.Declaration);
  if (flags & (ts.ModifierFlags.Private | ts.ModifierFlags.Protected)) return true;
  const name = (member as { name?: ts.PropertyName }).name;
  return name !== undefined && name.getText().startsWith("#");
}

/** Every exported item, and every public member of an exported class, interface, enum, or object type alias, that lacks a doc comment. */
function findUndocumented(): Missing[] {
  const missing: Missing[] = [];

  for (const file of readdirSync(SRC).filter((f) => f.endsWith(".ts")).sort()) {
    const text = readFileSync(new URL(file, SRC), "utf8");
    const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);

    const note = (node: ts.Node, name: string): void => {
      if (docOf(node) !== null) return;
      const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
      missing.push({ file, line: line + 1, name });
    };

    // A function, method, or constructor can be declared as several
    // signatures followed by one with a body, which implements them. A
    // caller sees only the bodyless signatures, so those are what need a
    // comment; the implementation is exempt.
    function checkOverloadAware(
      group: ReadonlyArray<{ body?: ts.Node } & ts.Node>,
      name: string,
    ): void {
      const overloaded = group.length > 1;
      for (const m of group) {
        if (overloaded && m.body !== undefined) continue;
        note(m, name);
      }
    }

    function classMembers(cls: ts.ClassDeclaration, className: string): void {
      const byName = new Map<string, ts.ClassElement[]>();
      for (const m of cls.members) {
        if (ts.isSemicolonClassElement(m)) continue;
        const key = ts.isConstructorDeclaration(m) ? "constructor" : m.name?.getText() ?? "(member)";
        const list = byName.get(key) ?? [];
        list.push(m);
        byName.set(key, list);
      }
      for (const [key, group] of byName) {
        const getters = group.filter(ts.isGetAccessor);
        const setters = group.filter(ts.isSetAccessor);
        if (getters.length > 0 || setters.length > 0) {
          // A getter and its setter are one item: either one's comment covers the pair.
          const pub = [...getters, ...setters].filter((m) => !isPrivateLike(m));
          if (pub.length === 0 || pub.some((m) => docOf(m) !== null)) continue;
          const { line } = sf.getLineAndCharacterOfPosition(pub[0]!.getStart(sf));
          missing.push({ file, line: line + 1, name: `${className}.${key}` });
          continue;
        }
        // A parameterless constructor takes nothing a caller needs explained.
        const checkable = group
          .filter((m) => !isPrivateLike(m))
          .filter((m) => !(ts.isConstructorDeclaration(m) && m.parameters.length === 0));
        checkOverloadAware(checkable as Array<{ body?: ts.Node } & ts.Node>, `${className}.${key}`);
      }
    }

    const topFunctions = new Map<string, ts.FunctionDeclaration[]>();
    ts.forEachChild(sf, (node) => {
      if (ts.isFunctionDeclaration(node) && node.name) {
        const list = topFunctions.get(node.name.text) ?? [];
        list.push(node);
        topFunctions.set(node.name.text, list);
      }
    });

    ts.forEachChild(sf, (node) => {
      if (ts.isFunctionDeclaration(node) && node.name && isExported(node)) {
        checkOverloadAware(topFunctions.get(node.name.text)!, node.name.text);
      } else if (ts.isClassDeclaration(node) && node.name && isExported(node)) {
        note(node, node.name.text);
        classMembers(node, node.name.text);
      } else if (ts.isInterfaceDeclaration(node) && isExported(node)) {
        note(node, node.name.text);
        for (const m of node.members) note(m, `${node.name.text}.${m.name?.getText() ?? "(member)"}`);
      } else if (ts.isTypeAliasDeclaration(node) && isExported(node)) {
        note(node, node.name.text);
        if (ts.isTypeLiteralNode(node.type)) {
          for (const m of node.type.members) note(m, `${node.name.text}.${m.name?.getText() ?? "(member)"}`);
        }
      } else if (ts.isEnumDeclaration(node) && isExported(node)) {
        note(node, node.name.text);
        for (const m of node.members) note(m, `${node.name.text}.${m.name.getText()}`);
      } else if (ts.isVariableStatement(node) && isExported(node)) {
        for (const decl of node.declarationList.declarations) note(node, decl.name.getText());
      }
    });
  }

  return missing;
}

test("every exported item in src carries a doc comment", () => {
  const missing = findUndocumented();
  const report = missing.map((m) => `${m.file}:${m.line}\t${m.name}`).join("\n");
  assert.equal(missing.length, 0, `${missing.length} item(s) without a doc comment:\n${report}`);
});
