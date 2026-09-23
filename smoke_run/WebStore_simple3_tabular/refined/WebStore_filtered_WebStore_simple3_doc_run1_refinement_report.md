# Stage 5 - Requirement Refinement Report

## Summary

| Outcome | Count |
|---|---:|
| Original requirements in the source document | 5 |
| Refined (kept) | 19 |
| Demoted to design constraints | 1 |
| Rejected | 3 |
| **Total requirements in the refined document** | **24** |

Numbering dialect: `dot`, major section `2`

## Kept requirements

- `2.6` The system shall create an Account only when the User provides a valid email address, password, and all required personal information; otherwise, the system shall display an error message and shall not create or store the Account.
  - view `dependency|operation`, gap `MISSING_EXCEPTION_HANDLING|MISSING_PRECONDITION|MISSING_VALIDATION`, sources FR-001, FR-004, 5 merged candidate(s)
- `2.7` The system shall allow an authenticated User to update the contact information, password, addresses, and Payment Preferences associated with the User's Account. The system shall validate the submitted data, persist valid changes, and return the updated account data with a confirmation message; if validation or storage fails, it shall display an error message and leave the Account unchanged.
  - view `dependency|operation`, gap `MISSING_CONSUMPTION|MISSING_PRODUCTION_FAILURE_HANDLING|MISSING_UPDATE|MISSING_VALIDATION`, sources FR-004, 4 merged candidate(s)
- `2.8` The system shall permit access to an Account's contact information, Payment Preferences, and Purchase History only to the authenticated User associated with that Account.
  - view `operation`, gap `MISSING_AUTHORIZATION`, sources FR-002, FR-004, 1 merged candidate(s)
- `2.9` The system shall make an authenticated User's stored Payment Preferences available when the User selects a payment method during checkout.
  - view `isolation`, gap `UNDER_SPECIFIED_DATA_USAGE`, sources FR-004, FR-014, 1 merged candidate(s)
- `2.10` The system shall permit only an authenticated Administrator to add, edit, or remove Products from the Inventory.
  - view `dependency`, gap `MISSING_AUTHORIZATION`, sources FR-006, 1 merged candidate(s)
- `2.11` The system shall reject an Administrator's attempt to add or edit a Product when required Product data is missing or invalid, or when the specified category is not one of the predefined Product Categories, and shall display an error message.
  - view `operation`, gap `MISSING_VALIDATION`, sources FR-005, FR-006, FR-007, 1 merged candidate(s)
- `2.12` The system shall validate Inventory Updates caused by Product administration, cart changes, or completed orders and shall prevent the Inventory Level of any Product from becoming negative, including when multiple updates occur concurrently.
  - view `operation`, gap `MISSING_VALIDATION`, sources FR-006, FR-008, FR-016, 1 merged candidate(s)
- `2.13` After an item is added to, modified in, or removed from a Shopping Cart, the system shall update the cart display so that its Product IDs, quantities, prices, and total cost reflect the resulting cart contents; for an authenticated User, the same resulting contents shall be saved between sessions.
  - view `dependency`, gap `MISSING_CONSISTENCY_HANDLING`, sources FR-009, FR-012, FR-010, FR-011, 2 merged candidate(s)
- `2.14` After a User attempts to add, modify, or remove a Shopping Cart item, the system shall display the updated cart and a confirmation message if the operation succeeds, or an error message if it fails.
  - view `dependency`, gap `MISSING_NOTIFICATION|MISSING_TRIGGERED_RESPONSE`, sources FR-009, FR-011, FR-010, FR-013, 2 merged candidate(s)
- `2.15` The system shall use the current contents of the User's Shopping Cart during checkout and shall complete an Order only when the Cart contains at least one item, a shipping address and supported payment method have been selected, and payment confirmation has been received; the system shall assign a unique Order ID to the completed Order.
  - view `dependency|operation`, gap `MISSING_CONSUMPTION|MISSING_VALIDATION`, sources FR-009, FR-013, FR-014, FR-015, FR-016, 2 merged candidate(s)
- `2.16` The system shall permit an authenticated User to add items to, view, modify, or remove items from only the Shopping Cart associated with that User's Account.
  - view `operation`, gap `MISSING_AUTHORIZATION`, sources FR-009, FR-010, FR-011, FR-012, 1 merged candidate(s)
- `2.17` The system shall reject a cart addition, quantity change, or removal when the Product is unknown or unavailable, the quantity is invalid or exceeds the available Inventory Level, or the modification command is malformed. For a rejected operation or persistence failure, the system shall display an error message, retain the Cart contents from before the failed operation, and not display a confirmation message.
  - view `operation`, gap `MISSING_EXCEPTION_HANDLING|MISSING_VALIDATION`, sources FR-009, FR-011, FR-010, FR-012, 2 merged candidate(s)
- `2.18` After payment has been processed and the related Inventory Update and Transaction recording have succeeded, the system shall generate an Order Confirmation and Receipt for the Order and send the User an Email Notification containing the Order Confirmation.
  - view `dependency|isolation`, gap `MISSING_CONSUMPTION|MISSING_VALIDATION|UNDERSPECIFIED_OUTPUT_FLOW`, sources FR-015, FR-016, 3 merged candidate(s)
- `2.19` The system shall report to the User when an Order Confirmation Email Notification cannot be generated or sent and shall not indicate that email confirmation succeeded unless the notification was successfully generated and sent.
  - view `dependency`, gap `MISSING_PRODUCTION_FAILURE_HANDLING`, sources FR-015, 2 merged candidate(s)
- `2.20` After payment is processed, the system shall record a Transaction and associate it with the corresponding Order, payment confirmation, and Inventory Update.
  - view `isolation`, gap `UNDER_SPECIFIED_DATA_DEPENDENCY`, sources FR-016, 1 merged candidate(s)
- `2.21` The system shall not report an Order as successfully completed or issue its Order Confirmation and Receipt until the unique Order ID has been generated, Inventory has been successfully updated, and the Transaction has been recorded. If an Inventory Update or Transaction recording fails, the system shall display an error message and shall not issue successful completion outputs.
  - view `operation`, gap `MISSING_EXCEPTION_HANDLING`, sources FR-008, FR-016, FR-013, FR-014, FR-015, 2 merged candidate(s)
- `2.22` The system shall permit Plugin installation, configuration, and management only when the User is authenticated as an Administrator.
  - view `isolation`, gap `UNDERSPECIFIED_AUTHORIZATION`, sources FR-019, 1 merged candidate(s)
- `2.23` When an Administrator successfully installs, configures, or manages a Plugin, the system shall apply the change to the affected Plugin-based functionality and record the operation in the Plugin administration log; if the operation fails, the system shall record the failure and display an error message.
  - view `dependency`, gap `MISSING_STATE_PROPAGATION`, sources FR-019, 1 merged candidate(s)
- `2.24` Before installing or configuring a Plugin, the system shall validate the supplied Plugin files and configuration settings and shall reject invalid or incompatible inputs with an error message.
  - view `operation`, gap `MISSING_VALIDATION`, sources FR-017, FR-019, 1 merged candidate(s)

## Demoted to design constraints

- If Transaction recording fails after payment has been processed, the system shall recover or retry the recording and Inventory synchronization without processing the payment again.
  - reason: The no-duplicate-payment recovery objective is useful engineering guidance, but the prescribed retry and recovery behavior is not established by FR-016.
  - merged candidates: candidate-46f9facc14f3cda5

## Rejected

- The system shall allow an authenticated User to view stored Account details, including contact information, Payment Preferences, and Purchase History.
  - reason: FR-004 requires storage but does not establish a user-facing operation for viewing all stored Account details.
  - merged candidates: candidate-413ece44f36478cc
- The system shall allow an authenticated User to delete the User's Account and provide a confirmation message when deletion is complete.
  - reason: Account deletion is not stated or otherwise supported by FR-001 through FR-004.
  - merged candidates: candidate-bdb8f34574854c09
- Following a failed Plugin installation, configuration, or initialization, the system shall restore the Plugin and its configuration settings to their pre-operation state.
  - reason: The required rollback semantics are not supported by FR-019; its supported logging and error behavior is already covered by another consolidated requirement.
  - merged candidates: candidate-6ecce2e6a081b153
