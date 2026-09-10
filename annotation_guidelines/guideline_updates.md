# Annotation Guideline Updates — Version 1.1

## Purpose

These updates are based on model error analysis and are intended
to improve annotation consistency for ambiguous customer messages.

---

## 1. Multi-Intent Messages

When a customer message contains multiple intents, assign the
single intent that represents the customer's primary request.

### Example

Customer:
"The item was returned, but I haven't received the refund."

Potential intents:
- return_request
- refund_status

Primary intent should be selected based on the customer's
main unresolved request.

If the message primarily asks about missing refund money,
use:

refund_status

If the message primarily asks to initiate or complete a return,
use:

return_request

If the primary intent cannot be determined confidently,
mark the conversation for review.

---

## 2. Damaged Item vs Missing Item

Use `damaged_item` when the expected product was received but
the product is physically damaged.

Use `missing_item` when an expected product or component was
not received.

### Example

"The phone arrived with a cracked screen."

→ damaged_item

"The phone was delivered, but the charger was missing."

→ missing_item

When wording contains both damage and missing-item information,
review the message carefully and identify the primary issue.

---

## 3. Duplicate Charge vs Payment Failure

Use `duplicate_charge` only when the customer reports being
charged more than once for the same transaction/order.

Examples:

"I was charged twice."

"I see two charges for the same purchase."

→ duplicate_charge

Use `payment_failure` when the customer cannot complete a
payment or the payment is rejected.

Examples:

"My payment was declined."

"My card isn't being accepted."

→ payment_failure

A failed or rejected payment should NOT be labeled
`duplicate_charge` unless the customer explicitly reports
multiple charges.

---

## 4. QA Feedback Rule

When model errors reveal a recurring ambiguity, the annotation
team should review the corresponding guideline and add
clarifying examples where necessary.

Guideline version: 1.1