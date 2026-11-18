# Is the Rosmaster hooked up right?

Two scripts: one runs **on the car**, one runs **on the laptop**. Together they check the car's board, the Wi-Fi link between them, and the data going each way. Neither one drives the car.

## 1. On the laptop: start the stack

From the repository folder:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Wait a minute, then:

```bash
python scripts/setup_entities.py
```

Find the laptop's IP address: run `ipconfig` and look for **IPv4 Address** under your Wi-Fi adapter, for example `192.168.1.50`. You need it on the car.

## 2. On the car: stop anything using the board

Only one program can talk to the board at a time. Stop the safety gate (`physical_ml_subscriber.py`) and Yahboom's ROS2 driver if either is running: press Ctrl+C in its terminal.

## 3. On the car: run the car check

The car needs this folder, plus `scripts/physical_ml_subscriber.py` copied **into** it (or a copy of the whole repository). Then, using the laptop's IP from step 1:

```bash
python3 check_rosmaster.py --broker 192.168.1.50
```

It asks you to **turn any wheel by hand** for 3 seconds. Then it listens for 60 seconds, so do step 4 straight away.

## 4. On the laptop: run the laptop check

While the car check is still listening:

```bash
python scripts/rosmaster_check/check_lab_link.py --no-pose
```

Leave out `--no-pose` once the overhead camera is sending the car's position.

## 5. Read the results

Each line starts with one word:

| Word | Means |
| --- | --- |
| **PASS** | Good |
| **WARN** | It works, but read the hint. A WARN saying **"unverified"** means our script couldn't read that part of the board. Tell whoever looks after the scripts; it doesn't mean the car is broken |
| **FAIL** | Something is wrong. The hint after it says what to try |
| **SKIP** | Not checked this time, and the hint says why |

**What a good run looks like:** the laptop shows PASS for Ditto, broker and link, and the car shows PASS for board, battery and broker.

## 6. Afterwards: start the safety gate again

With the gate stopped, nothing can cut the car's motors in an emergency. **A pass does not mean the car is ready to drive.** On the car:

```bash
python3 physical_ml_subscriber.py --broker 192.168.1.50 --driver rosmaster
```

## Common problems

| You see | Try |
| --- | --- |
| `broker: can't reach ...` | The car and laptop must be on the same Wi-Fi. Check the IP you typed, and that step 1 worked |
| `board: couldn't open it ... busy` or `... denied` | Something else has the board: stop the safety gate and the ROS2 driver (step 2) |
| `board: ... Rosmaster_Lib isn't installed` | You're not on the car, or Yahboom's library isn't installed on it |
| `link: no replies` | The car check isn't running, or has stopped listening: start step 3 again, then step 4 straight away |
| `link: FAIL ... would hold it` | The Wi-Fi is too slow or drops messages. Move closer to the router, or turn off other heavy Wi-Fi use |
| `battery: ... charge it` | Charge the battery |
| `encoders: no wheel moved` | You didn't turn a wheel in time, or an encoder cable is loose. Run step 3 again and turn a wheel when it says |
| `Ditto: no answer` or `there's no vehicle_ml Thing` | Do step 1 again |

## Testing the motors (optional)

This spins the wheels for 1 second. **Lift the car so the wheels are off the ground first.** It asks you to type `yes`.

```bash
python3 check_rosmaster.py --broker 192.168.1.50 --motors
```
