# Tracking System Interface

## Coordinate System

### Image Coordinates

Origin: top-left

X → right

Y → down

---

## Simulator → Tracking System

### get_frame()

Returns:

- 640 × 480 image frame

---

### get_ground_truth()

Returns:

```python
{
    "x": float,
    "y": float
}